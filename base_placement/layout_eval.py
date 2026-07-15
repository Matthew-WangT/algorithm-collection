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
from .robot_model import DEFAULT_URDF, ArmModel, make_base_pose
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
                 urdf_path: str = DEFAULT_URDF):
        self.cmaps = {"left": cmap_left, "right": cmap_right}
        # urdf_path 必须显式贯通：胶囊链 FK 用的 ArmModel 要与建图/精评同一 URDF，
        # 否则会静默回退到 DEFAULT_URDF（换机器/换标定 URDF 时打分模型错配）。
        self.arms = {"left": ArmModel("left", urdf_path),
                     "right": ArmModel("right", urdf_path)}
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
        p = self.p
        T_W = {s: make_base_pose(s, d, theta, p.x_base, p.z_base, p.pitch)
                  .homogeneous
               for s in ("left", "right")}

        eL = self._eval_arm("left", T_W["left"], self.tasks.left)
        eR = self._eval_arm("right", T_W["right"], self.tasks.right)
        sL = self._eval_arm("left", T_W["left"], self.tasks.shared)
        sR = self._eval_arm("right", T_W["right"], self.tasks.shared)

        # 臂间自碰撞初筛：left[i] 与 right[i] 配对（双方都可达才有意义）
        n_pair = min(len(self.tasks.left), len(self.tasks.right))
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

        qual_L = float(np.average(eL["quality"], weights=self.tasks.w_left))
        qual_R = float(np.average(eR["quality"], weights=self.tasks.w_right))
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


def _smoke():
    from .task_points import synthetic_taskset
    cl = CapabilityMap.load("out_base_placement/maps/capmap_left.npz")
    cr = CapabilityMap.load("out_base_placement/maps/capmap_right.npz")
    ev = LayoutEvaluator(cl, cr, synthetic_taskset())
    import time
    t0 = time.time()
    r = ev.score_layout(0.6, 0.5)
    print(f"单布局耗时 {time.time()-t0:.2f}s")
    for k, v in r.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    _smoke()
