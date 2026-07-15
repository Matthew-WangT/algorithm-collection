#!/usr/bin/env python3
"""Top-k 布局的直接 IK 精评（决赛圈）。

对粗扫描选出的每个布局：逐任务点做 DLS 数值 IK（多随机初值 + 能力图代表 q
warm-start），成功解里取可操作度最高者，完整计算：
- 可达性（IK 成功且满足限位）
- 归一化可操作度 w_norm、位置块条件数 kappa
- 桌面裕度、臂间自碰撞（配对，同 layout_eval 约定）

作用：修正查表阶段的两类偏差——FK 采样图的保守漏报、bin 代表 q 的
悲观碰撞判定；产出最终排序与逐点诊断数据。
"""

from __future__ import annotations

import multiprocessing as mp

import numpy as np
import pinocchio as pin

from .capsules import (DEFAULT_RADII, capsule_capsule_min_distance,
                       chain_table_margin, transform_chain)
from .robot_model import (DEFAULT_BASE_FRAME, DEFAULT_EE_FRAME,
                          DEFAULT_JOINT_PREFIX, ArmModel, make_base_pose)
from .task_points import TaskSet


def dls_ik(arm: ArmModel, T_B_target: pin.SE3, q0: np.ndarray,
           tol_pos: float = 1e-3, tol_rot: float = 1e-2,
           max_iter: int = 100, damping: float = 1e-6) -> tuple[bool, np.ndarray]:
    """阻尼最小二乘 IK（臂 base 系目标）。返回 (成功, q)。

    误差用 SE3 对数映射（LOCAL 系），步长自适应缩放，限位硬钳位。
    """
    q = q0.copy()
    for _ in range(max_iter):
        T_cur = arm.fk_base(q)
        err = pin.log(T_cur.actInv(T_B_target)).vector  # LOCAL 系 6 维
        if np.linalg.norm(err[:3]) < tol_pos and np.linalg.norm(err[3:]) < tol_rot:
            return True, q
        q_full = arm.full_q(q)
        pin.forwardKinematics(arm.model, arm.data, q_full)
        J = pin.computeFrameJacobian(
            arm.model, arm.data, q_full, arm.ee_frame_id,
            pin.ReferenceFrame.LOCAL)[:, arm.v_indices]
        JJt = J @ J.T
        lam = damping + 1e-4 * np.trace(JJt) / 6.0
        dq = J.T @ np.linalg.solve(JJt + lam * np.eye(6), err)
        step = min(1.0, 0.5 / max(np.abs(dq).max(), 1e-9))
        q = np.clip(q + step * dq, arm.lower, arm.upper)
    # 最终再判一次
    err = pin.log(arm.fk_base(q).actInv(T_B_target)).vector
    ok = np.linalg.norm(err[:3]) < tol_pos and np.linalg.norm(err[3:]) < tol_rot
    return ok, q


def solve_point(arm: ArmModel, T_B_target: pin.SE3, q_warm: np.ndarray | None,
                n_seed: int, rng: np.random.Generator,
                tol_pos: float, tol_rot: float, max_iter: int
                ) -> tuple[bool, np.ndarray | None, float]:
    """多初值求解，返回 (可达, 最佳 q, 最佳 w)。最佳 = 成功解中 w 最大。"""
    seeds = []
    if q_warm is not None and np.any(q_warm):
        seeds.append(np.clip(q_warm, arm.lower, arm.upper))
    seeds += list(arm.sample_arm_q(n_seed, rng))
    best_q, best_w = None, -1.0
    for q0 in seeds:
        ok, q = dls_ik(arm, T_B_target, q0, tol_pos, tol_rot, max_iter)
        if ok:
            _, w = arm.fk_and_manip(q)
            if w > best_w:
                best_q, best_w = q, w
    return best_q is not None, best_q, best_w


# ---------------- 并行 worker ----------------

_G: dict = {}


def _init_worker(urdf_path: str, base_frame: str, ee_frame: str,
                 joint_prefix: str):
    _G["left"] = ArmModel("left", urdf_path, base_frame, ee_frame, joint_prefix)
    _G["right"] = ArmModel("right", urdf_path, base_frame, ee_frame, joint_prefix)


def _solve_task(args):
    (side, T_B_flat, q_warm, n_seed, seed, tol_pos, tol_rot, max_iter) = args
    arm: ArmModel = _G[side]
    T = pin.SE3(np.asarray(T_B_flat).reshape(4, 4))
    rng = np.random.default_rng(seed)
    ok, q, w = solve_point(arm, T, q_warm, n_seed, rng, tol_pos, tol_rot, max_iter)
    if not ok:
        return False, np.zeros(7), 0.0, 0.0, np.zeros((9, 3))
    kappa = arm.jacobian_pos_cond(q)
    chain_B = arm.joint_positions_base(q)
    return True, q, w, kappa, chain_B


def refine_layout(d: float, theta: float, tasks: TaskSet, w_ref: dict,
                  urdf_path: str, z_table: float = 0.75,
                  x_base: float = 0.0, z_base: float = 1.4, pitch: float = 1.57,
                  q_warms: dict | None = None,
                  n_seed: int = 10, tol_pos: float = 1e-3, tol_rot: float = 1e-2,
                  max_iter: int = 100, d_self_safe: float = 0.05,
                  kappa_max: float = 50.0,
                  base_frame: str = DEFAULT_BASE_FRAME,
                  ee_frame: str = DEFAULT_EE_FRAME,
                  joint_prefix: str = DEFAULT_JOINT_PREFIX,
                  n_workers: int = 16, seed: int = 7) -> dict:
    """对单个布局做直接 IK 全评估。

    w_ref: {'left': w98, 'right': w98} 能力图归一化基准（与粗扫一致口径）。
    q_warms: {'left': (N,7), 'right': (N,7), 'shared_left': ..., 'shared_right': ...}
             查表得到的代表 q（可为 None）。
    返回逐点结果与汇总指标。
    """
    T_W = {s: make_base_pose(s, d, theta, x_base, z_base, pitch).homogeneous
           for s in ("left", "right")}
    inv_W = {s: np.linalg.inv(T_W[s]) for s in T_W}

    jobs, meta = [], []
    groups = [("left", "left", tasks.left), ("right", "right", tasks.right),
              ("shared_left", "left", tasks.shared),
              ("shared_right", "right", tasks.shared)]
    for gname, side, targets in groups:
        for i, T_t in enumerate(targets):
            T_B = inv_W[side] @ T_t
            qw = None
            if q_warms and gname in q_warms and q_warms[gname] is not None:
                qw = q_warms[gname][i]
            jobs.append((side, T_B.ravel(), qw, n_seed,
                         seed + len(jobs), tol_pos, tol_rot, max_iter))
            meta.append((gname, i))

    with mp.Pool(n_workers, initializer=_init_worker,
                 initargs=(urdf_path, base_frame, ee_frame,
                           joint_prefix)) as pool:
        outs = pool.map(_solve_task, jobs, chunksize=8)

    res = {g: {"reachable": [], "q": [], "w_norm": [], "kappa": [],
               "margin": [], "chain_W": []}
           for g, _, _ in groups}
    for (gname, i), (ok, q, w, kappa, chain_B) in zip(meta, outs):
        side = "left" if "left" in gname else "right"
        r = res[gname]
        r["reachable"].append(ok)
        r["q"].append(q)
        r["kappa"].append(kappa if ok else np.nan)
        r["w_norm"].append(min(w / w_ref[side], 1.0) if ok else 0.0)
        if ok:
            chain_W = transform_chain(T_W[side], chain_B)
            r["chain_W"].append(chain_W)
            r["margin"].append(chain_table_margin(chain_W, z_table))
        else:
            r["chain_W"].append(np.full((9, 3), np.nan))
            r["margin"].append(np.nan)
    for r in res.values():
        for k in r:
            r[k] = np.array(r[k])

    # 配对臂间碰撞
    n_pair = min(len(tasks.left), len(tasks.right))
    both = res["left"]["reachable"][:n_pair] & res["right"]["reachable"][:n_pair]
    pair_dist = np.full(n_pair, np.nan)
    for i in np.flatnonzero(both):
        pair_dist[i] = capsule_capsule_min_distance(
            res["left"]["chain_W"][i], res["right"]["chain_W"][i],
            DEFAULT_RADII, DEFAULT_RADII)
    coll_rate = float((pair_dist[both] < d_self_safe).mean()) if both.any() else 0.0

    # 汇总（与 layout_eval 同口径 + 条件数惩罚率）
    summary = {"d": d, "theta": theta, "collision_rate": coll_rate}
    for gname in ("left", "right"):
        r = res[gname]
        summary[f"reach_rate_{gname}"] = float(r["reachable"].mean())
        summary[f"mean_w_{gname}"] = (
            float(r["w_norm"][r["reachable"]].mean()) if r["reachable"].any() else 0.0)
        summary[f"kappa_bad_rate_{gname}"] = (
            float((r["kappa"][r["reachable"]] > kappa_max).mean())
            if r["reachable"].any() else 0.0)
    both_s = res["shared_left"]["reachable"] & res["shared_right"]["reachable"]
    summary["shared_both_reach"] = float(both_s.mean()) if len(both_s) else 0.0
    summary["mean_table_margin"] = float(np.nanmean(np.concatenate(
        [res["left"]["margin"], res["right"]["margin"]])))
    summary["reach_rate"] = float(np.concatenate(
        [res["left"]["reachable"], res["right"]["reachable"]]).mean())
    summary["pair_dist"] = pair_dist
    return {"summary": summary, "points": res, "T_W": T_W}


if __name__ == "__main__":
    # 冒烟：单布局少量点（URDF/帧命名从 config.yaml 读取）
    from .robot_model import arm_kwargs_from_cfg, load_robot_cfg
    from .task_points import synthetic_taskset
    akw = arm_kwargs_from_cfg(load_robot_cfg())
    ts = synthetic_taskset(n_per_arm=20, n_shared=10)
    out = refine_layout(0.6, 0.5, ts, {"left": 0.1561, "right": 0.1561},
                        akw["urdf_path"], base_frame=akw["base_frame"],
                        ee_frame=akw["ee_frame"], joint_prefix=akw["joint_prefix"],
                        n_workers=8)
    for k, v in out["summary"].items():
        if not isinstance(v, np.ndarray):
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
