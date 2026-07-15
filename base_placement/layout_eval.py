#!/usr/bin/env python3
"""布局评分：给定 (d, theta) 用能力图查表打分（IRM 形态 A）。

流程（每个布局）：
1. make_base_pose 生成左右 T_W_B。
2. 任务点变换到各臂 base 系，姿态感知查能力图 → reachable / w_norm / q_repr。
3. 用 q_repr 做 FK 得胶囊链 → 桌面裕度、臂间自碰撞初筛（left[i]-right[i] 配对）。
4. 汇总分数与分项指标。

quality_i = reachable ? alpha*w_norm + beta*clip(margin/d_ref, 0, 1) : 0
Score = mean_i(quality_L) + mean_i(quality_R) - lambda_self*collision_rate
        (+ lambda_overlap*overlap_quality, 默认权重 0，仅输出分项)

条件数惩罚推迟到 top-k 直接 IK 精评阶段（查表阶段无精确雅可比）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .capability_map import CapabilityMap
from .capsules import (DEFAULT_RADII, capsule_capsule_min_distance,
                       chain_table_margin, transform_chain)
from .robot_model import (DEFAULT_BASE_FRAME, DEFAULT_EE_FRAME,
                          DEFAULT_JOINT_PREFIX, ArmModel, make_base_pose)
from .task_points import TaskSet


@dataclass
class ScoringParams:
    alpha_manip: float = 1.0
    beta_table: float = 0.5
    d_ref_table: float = 0.15
    lambda_self: float = 2.0
    lambda_overlap: float = 0.0     # 拷问决策：overlap 输出分项但默认不进总分
    d_self_safe: float = 0.05
    w_overlap_min: float = 0.2      # overlap 里 "高质量" 的 w_norm 门槛
    z_table: float = 0.75
    x_base: float = 0.0
    z_base: float = 1.4
    pitch: float = 1.57


class LayoutEvaluator:
    def __init__(self, cmap_left: CapabilityMap, cmap_right: CapabilityMap,
                 tasks: TaskSet, params: ScoringParams | None = None,
                 urdf_path: str | None = None,
                 base_frame: str = DEFAULT_BASE_FRAME,
                 ee_frame: str = DEFAULT_EE_FRAME,
                 joint_prefix: str = DEFAULT_JOINT_PREFIX):
        if urdf_path is None:
            raise ValueError("LayoutEvaluator 需要 urdf_path（从 config.yaml 传入）")
        self.cmaps = {"left": cmap_left, "right": cmap_right}
        # URDF/帧命名必须显式贯通：胶囊链 FK 用的 ArmModel 要与建图/精评同一模型，
        # 否则打分模型与能力图错配。命名默认对应 rizon4，换机器人由 config 覆盖。
        self.arms = {
            "left": ArmModel("left", urdf_path, base_frame, ee_frame, joint_prefix),
            "right": ArmModel("right", urdf_path, base_frame, ee_frame, joint_prefix)}
        self.tasks = tasks
        self.p = params or ScoringParams()

    # ---------- 单臂一批任务点的查表评估 ----------

    def _eval_arm(self, side: str, T_W_B: np.ndarray, targets_W: np.ndarray) -> dict:
        """targets_W: (n,4,4) 世界系任务位姿。返回逐点指标（含胶囊链，供碰撞用）。"""
        inv_B = np.linalg.inv(T_W_B)
        T_B = np.einsum("ij,njk->nik", inv_B, targets_W)
        res = self.cmaps[side].query(T_B[:, :3, 3], T_B[:, :3, :3])

        n = len(targets_W)
        arm = self.arms[side]
        margins = np.full(n, np.nan)
        chains_W = np.full((n, 9, 3), np.nan)
        for i in np.flatnonzero(res["reachable"]):
            chain_B = arm.joint_positions_base(res["q_repr"][i])
            chain_W = transform_chain(T_W_B, chain_B)
            chains_W[i] = chain_W
            margins[i] = chain_table_margin(chain_W, self.p.z_table)

        table_term = np.where(
            res["reachable"] & (margins > 0),
            np.clip(np.nan_to_num(margins) / self.p.d_ref_table, 0.0, 1.0), 0.0)
        quality = np.where(res["reachable"],
                           self.p.alpha_manip * res["w_norm"]
                           + self.p.beta_table * table_term, 0.0)
        return {"reachable": res["reachable"], "w_norm": res["w_norm"],
                "margin": margins, "quality": quality, "chains_W": chains_W}

    # ---------- 布局评分 ----------

    def score_layout(self, d: float, theta: float) -> dict:
        """用构造时传入的 self.tasks 打分（原有 (d,θ) 扫描入口，行为不变）。"""
        return self.score_layout_with_tasks(d, theta, self.tasks)

    def score_layout_with_tasks(self, d: float, theta: float,
                                tasks: TaskSet) -> dict:
        """对指定任务集打分：score_layout 的主体，任务集参数化。

        供"操作区中心 p 搜索"复用——布局固定、任务集平移时，
        变换到臂 base 系之后的查表/碰撞/打分逻辑完全同构。
        """
        p = self.p
        T_W = {s: make_base_pose(s, d, theta, p.x_base, p.z_base, p.pitch)
                  .homogeneous
               for s in ("left", "right")}

        eL = self._eval_arm("left", T_W["left"], tasks.left)
        eR = self._eval_arm("right", T_W["right"], tasks.right)
        sL = self._eval_arm("left", T_W["left"], tasks.shared)
        sR = self._eval_arm("right", T_W["right"], tasks.shared)

        # 臂间自碰撞初筛：left[i] 与 right[i] 配对（双方都可达才有意义）
        n_pair = min(len(tasks.left), len(tasks.right))
        both = eL["reachable"][:n_pair] & eR["reachable"][:n_pair]
        n_coll = 0
        pair_dists = np.full(n_pair, np.nan)
        for i in np.flatnonzero(both):
            dmin = capsule_capsule_min_distance(
                eL["chains_W"][i], eR["chains_W"][i], DEFAULT_RADII, DEFAULT_RADII)
            pair_dists[i] = dmin
            n_coll += int(dmin < p.d_self_safe)
        coll_rate = n_coll / max(int(both.sum()), 1)

        # overlap：共享区左右都可达且都不差的比例
        both_s = sL["reachable"] & sR["reachable"]
        hq = both_s & (sL["w_norm"] >= p.w_overlap_min) \
                    & (sR["w_norm"] >= p.w_overlap_min)
        overlap_quality = float(np.average(hq)) if len(hq) else 0.0

        qual_L = float(np.average(eL["quality"], weights=tasks.w_left))
        qual_R = float(np.average(eR["quality"], weights=tasks.w_right))
        score = (qual_L + qual_R
                 - p.lambda_self * coll_rate
                 + p.lambda_overlap * overlap_quality)

        return {
            "d": d, "theta": theta, "score": score,
            "quality_left": qual_L, "quality_right": qual_R,
            "reach_rate": float(np.concatenate(
                [eL["reachable"], eR["reachable"]]).mean()),
            "mean_w": float(np.concatenate(
                [eL["w_norm"][eL["reachable"]],
                 eR["w_norm"][eR["reachable"]]]).mean()
                if (eL["reachable"].any() or eR["reachable"].any()) else 0.0),
            "collision_rate": float(coll_rate),
            "overlap_quality": overlap_quality,
            "mean_table_margin": float(np.nanmean(np.concatenate(
                [eL["margin"], eR["margin"]]))
                if np.isfinite(np.concatenate(
                    [eL["margin"], eR["margin"]])).any() else np.nan),
            "shared_both_reach": float(both_s.mean()) if len(both_s) else 0.0,
        }

    def scan_grid(self, d_values: np.ndarray, theta_values: np.ndarray,
                  verbose: bool = True) -> list[dict]:
        results = []
        total = len(d_values) * len(theta_values)
        for a, d in enumerate(d_values):
            for b, th in enumerate(theta_values):
                r = self.score_layout(float(d), float(th))
                results.append(r)
                if verbose:
                    k = a * len(theta_values) + b + 1
                    print(f"[{k}/{total}] d={d:.2f} θ={th:.2f} "
                          f"score={r['score']:.3f} reach={r['reach_rate']:.1%} "
                          f"coll={r['collision_rate']:.1%}")
        return results

    def scan_region_grid(self, d_fixed: float, theta_fixed: float,
                         base_tasks: TaskSet,
                         px_values: np.ndarray, pz_values: np.ndarray,
                         z_offset: float = 0.0,
                         verbose: bool = True) -> list[dict]:
        """布局 (d,θ) 固定，扫描操作区中心 p=(px, pz) 网格。

        base_tasks 为局部系任务集（centered_taskset，AABB 中心为原点、
        桌面 z=0），每个候选做刚性平移 [px, 0, z_offset + pz] 后打分。
        py 恒为 0：双臂沿 xz 平面对称布置且合成任务集左右对称，Score 关于
        py=0 镜像对称，扫 y 无信息量（真实非对称任务接入后再议）。
        z_offset 传 z_table：pz 是相对"桌面基准高度"的偏移，pz=0 即现状。
        pz_values 为单元素时自然退化为纯 x 的 1D 搜索，无需特判。
        """
        results = []
        total = len(px_values) * len(pz_values)
        for a, px in enumerate(px_values):
            for b, pz in enumerate(pz_values):
                tasks_p = base_tasks.translated(
                    [float(px), 0.0, z_offset + float(pz)])
                r = self.score_layout_with_tasks(d_fixed, theta_fixed, tasks_p)
                r["px"], r["pz"] = float(px), float(pz)
                results.append(r)
                if verbose:
                    k = a * len(pz_values) + b + 1
                    print(f"[{k}/{total}] px={px:.2f} pz={pz:.2f} "
                          f"score={r['score']:.3f} reach={r['reach_rate']:.1%} "
                          f"coll={r['collision_rate']:.1%}")
        return results


def _smoke():
    from .robot_model import arm_kwargs_from_cfg, load_robot_cfg
    from .task_points import centered_taskset, synthetic_taskset
    cl = CapabilityMap.load("out_base_placement/maps/capmap_left.npz")
    cr = CapabilityMap.load("out_base_placement/maps/capmap_right.npz")
    akw = arm_kwargs_from_cfg(load_robot_cfg())
    ev = LayoutEvaluator(cl, cr, synthetic_taskset(), **akw)
    import time
    t0 = time.time()
    r = ev.score_layout(0.6, 0.5)
    print(f"单布局耗时 {time.time()-t0:.2f}s")
    for k, v in r.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # region 冒烟：小任务集 + 3×2 的 (px,pz) 网格
    base = centered_taskset(n_per_arm=20, n_shared=10)
    rr = ev.scan_region_grid(0.6, 0.5, base,
                             np.linspace(0.5, 1.1, 3), np.array([0.0, 0.2]),
                             z_offset=0.75)
    assert len(rr) == 6 and all("px" in x and "pz" in x for x in rr)
    best = max(rr, key=lambda x: x["score"])
    print(f"region 冒烟 best: px={best['px']:.2f} pz={best['pz']:.2f} "
          f"score={best['score']:.3f}")


if __name__ == "__main__":
    _smoke()
