#!/usr/bin/env python3
"""端到端管线：build-map → scan → refine → report。

用法（retarget_nn 环境）：
  python -m base_placement.run_pipeline all            # 全流程
  python -m base_placement.run_pipeline build-map      # 仅建能力图
  python -m base_placement.run_pipeline scan           # 粗网格扫描 + 热力图
  python -m base_placement.run_pipeline refine         # top-k 直接 IK 精评 + 报告
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import yaml

PKG_DIR = os.path.dirname(os.path.abspath(__file__))


def load_cfg(path: str | None) -> dict:
    with open(path or os.path.join(PKG_DIR, "config.yaml")) as f:
        return yaml.safe_load(f)


def map_path(cfg, side):
    return os.path.join(cfg["map"]["dir"], f"capmap_{side}.npz")


def cmd_task_aabb(cfg, cfg_path, hdf5_root, max_files=None, margin=0.0):
    """用真实 hdf5 采集数据里的 eef AABB 更新 config.yaml 的 task 区间。"""
    from .hdf5_eef_workspace import scan_workspace

    result = scan_workspace(hdf5_root, max_files=max_files, arms="both")
    left_aabb, right_aabb = result["left_aabb"], result["right_aabb"]
    if left_aabb is None or right_aabb is None:
        sys.exit("左/右臂 AABB 数据不足，无法更新 task 配置")

    z_table = cfg["task"]["z_table"]
    x_lo = min(left_aabb["min"][0], right_aabb["min"][0]) - margin
    x_hi = max(left_aabb["max"][0], right_aabb["max"][0]) + margin
    y_left = [left_aabb["min"][1] - margin, left_aabb["max"][1] + margin]
    y_right = [right_aabb["min"][1] - margin, right_aabb["max"][1] + margin]
    z_lo = min(left_aabb["min"][2], right_aabb["min"][2])
    z_hi = max(left_aabb["max"][2], right_aabb["max"][2])
    z_above = [z_lo - z_table - margin, z_hi - z_table + margin]

    from ruamel.yaml import YAML
    from ruamel.yaml.comments import CommentedSeq
    yaml_rt = YAML()
    yaml_rt.width = 4096  # 避免把 urdf_path 等无关长字符串行折行

    def _flow_seq(values):
        seq = CommentedSeq(values)
        seq.fa.set_flow_style()
        return seq

    with open(cfg_path) as f:
        doc = yaml_rt.load(f)
    doc["task"]["x_range"] = _flow_seq([round(x_lo, 4), round(x_hi, 4)])
    doc["task"]["y_left"] = _flow_seq([round(y_left[0], 4), round(y_left[1], 4)])
    doc["task"]["y_right"] = _flow_seq([round(y_right[0], 4), round(y_right[1], 4)])
    doc["task"]["z_above"] = _flow_seq([round(z_above[0], 4), round(z_above[1], 4)])
    with open(cfg_path, "w") as f:
        yaml_rt.dump(doc, f)
    print(f"已用 {hdf5_root} 的真实 eef AABB 更新 {cfg_path} 的 task 区间：\n"
          f"  x_range={doc['task']['x_range']} y_left={doc['task']['y_left']} "
          f"y_right={doc['task']['y_right']} z_above={doc['task']['z_above']}")


def _arm_kwargs(cfg):
    """从 config 的 robot 段取 ArmModel 关键字参数（urdf_path + 帧/关节命名）。"""
    from .robot_model import arm_kwargs_from_cfg
    return arm_kwargs_from_cfg(cfg["robot"])


def cmd_build_map(cfg):
    from .capability_map import build_capability_map, self_check
    from .robot_model import ArmModel
    m = cfg["map"]
    akw = _arm_kwargs(cfg)
    os.makedirs(m["dir"], exist_ok=True)
    for side in ("left", "right"):
        cm = build_capability_map(
            side, n_samples=int(m["n_fk_samples"]), voxel_size=m["voxel_size"],
            n_dirs=m["n_dirs"], n_rolls=m["n_rolls"], n_workers=m["n_workers"],
            **akw)
        self_check(cm, ArmModel(side, **akw))
        cm.save(map_path(cfg, side))
        print(f"saved {map_path(cfg, side)}")


def _load_maps_tasks(cfg):
    from .capability_map import CapabilityMap
    from .task_points import synthetic_taskset
    cl = CapabilityMap.load(map_path(cfg, "left"))
    cr = CapabilityMap.load(map_path(cfg, "right"))
    t = cfg["task"]
    ts = synthetic_taskset(
        z_table=t["z_table"], x_range=tuple(t["x_range"]),
        y_left=tuple(t["y_left"]), y_right=tuple(t["y_right"]),
        y_shared=tuple(t["y_shared"]), z_above=tuple(t["z_above"]),
        n_per_arm=t["n_per_arm"], n_shared=t["n_shared"], seed=t["seed"])
    return cl, cr, ts


def _scoring_params(cfg):
    from .layout_eval import ScoringParams
    s, L, t = cfg["scoring"], cfg["layout"], cfg["task"]
    return ScoringParams(
        alpha_manip=s["alpha_manip"], beta_table=s["beta_table"],
        d_ref_table=s["d_ref_table"], lambda_self=s["lambda_self"],
        lambda_overlap=s["lambda_overlap"], d_self_safe=s["d_self_safe"],
        w_overlap_min=s["w_overlap_min"], z_table=t["z_table"],
        x_base=L["x_base"], z_base=L["z_base"], pitch=L["pitch"])


def _grid(cfg):
    L = cfg["layout"]
    return (np.linspace(*L["d_range"], L["grid_d"]),
            np.linspace(*L["theta_range"], L["grid_theta"]))


def cmd_scan(cfg):
    from .layout_eval import LayoutEvaluator
    from .visualize import plot_capmap_slices, plot_scan_heatmaps
    cl, cr, ts = _load_maps_tasks(cfg)
    out_dir = cfg["output"]["dir"]
    figs = os.path.join(out_dir, "figs")
    os.makedirs(figs, exist_ok=True)

    plot_capmap_slices(cl, os.path.join(figs, "capmap_left_D.png"))
    plot_capmap_slices(cr, os.path.join(figs, "capmap_right_D.png"))

    ev = LayoutEvaluator(cl, cr, ts, _scoring_params(cfg), **_arm_kwargs(cfg))
    d_vals, th_vals = _grid(cfg)
    results = ev.scan_grid(d_vals, th_vals)
    with open(os.path.join(out_dir, "scan_results.json"), "w") as f:
        json.dump(results, f, indent=1)
    plot_scan_heatmaps(results, d_vals, th_vals,
                       os.path.join(figs, "scan_heatmaps.png"))
    best = max(results, key=lambda r: r["score"])
    print(f"scan best: d={best['d']:.2f} theta={best['theta']:.2f} "
          f"score={best['score']:.3f}")
    return results


def cmd_refine(cfg):
    from .capability_map import CapabilityMap
    from .layout_eval import LayoutEvaluator
    from .refine_ik import refine_layout
    from .visualize import plot_layout_3d

    out_dir = cfg["output"]["dir"]
    figs = os.path.join(out_dir, "figs")
    os.makedirs(figs, exist_ok=True)
    with open(os.path.join(out_dir, "scan_results.json")) as f:
        results = json.load(f)
    results.sort(key=lambda r: r["score"], reverse=True)
    top = results[: cfg["refine"]["top_k"]]

    cl, cr, ts = _load_maps_tasks(cfg)
    ev = LayoutEvaluator(cl, cr, ts, _scoring_params(cfg), **_arm_kwargs(cfg))
    r_cfg, s_cfg, L = cfg["refine"], cfg["scoring"], cfg["layout"]
    w_ref = {"left": cl.w98, "right": cr.w98}

    refined = []
    for k, cand in enumerate(top):
        d, th = cand["d"], cand["theta"]
        # 查表代表 q 作 warm-start
        q_warms = {}
        for gname, side, targets in [
                ("left", "left", ts.left), ("right", "right", ts.right),
                ("shared_left", "left", ts.shared),
                ("shared_right", "right", ts.shared)]:
            from .robot_model import make_base_pose
            T_W = make_base_pose(side, d, th, L["x_base"], L["z_base"],
                                 L["pitch"]).homogeneous
            T_B = np.einsum("ij,njk->nik", np.linalg.inv(T_W), targets)
            cm = cl if side == "left" else cr
            q_warms[gname] = cm.query(T_B[:, :3, 3], T_B[:, :3, :3])["q_repr"]

        akw = _arm_kwargs(cfg)
        out = refine_layout(
            d, th, ts, w_ref, akw["urdf_path"],
            z_table=cfg["task"]["z_table"], x_base=L["x_base"],
            z_base=L["z_base"], pitch=L["pitch"], q_warms=q_warms,
            n_seed=r_cfg["n_seed"], tol_pos=r_cfg["tol_pos"],
            tol_rot=r_cfg["tol_rot"], max_iter=r_cfg["max_iter"],
            d_self_safe=s_cfg["d_self_safe"], kappa_max=r_cfg["kappa_max"],
            base_frame=akw["base_frame"], ee_frame=akw["ee_frame"],
            joint_prefix=akw["joint_prefix"],
            n_workers=r_cfg["n_workers"], seed=100 + k)
        s = out["summary"]
        # 精评总分（与粗扫同结构，含条件数惩罚）
        qual = (s["mean_w_left"] * s["reach_rate_left"]
                + s["mean_w_right"] * s["reach_rate_right"])
        s["score_refined"] = (
            qual - s_cfg["lambda_self"] * s["collision_rate"]
            - 0.5 * (s["kappa_bad_rate_left"] + s["kappa_bad_rate_right"]))
        s["score_scan"] = cand["score"]
        refined.append((s, out))
        print(f"[refine {k+1}/{len(top)}] d={d:.2f} θ={th:.2f} "
              f"reach={s['reach_rate']:.1%} coll={s['collision_rate']:.1%} "
              f"score_refined={s['score_refined']:.3f}")

    refined.sort(key=lambda x: x[0]["score_refined"], reverse=True)
    best_s, best_out = refined[0]

    # 交叉验证：查表可达性 vs IK 可达性 一致率（最优布局）
    scan_best = ev.score_layout(best_s["d"], best_s["theta"])
    agree = {
        "scan_reach_rate": scan_best["reach_rate"],
        "ik_reach_rate": best_s["reach_rate"],
        "note": "FK 采样图为保守估计，scan ≤ ik 为正常方向",
    }

    plot_layout_3d(best_out, ts, cfg["task"]["z_table"],
                   os.path.join(figs, "best_layout_3d.png"),
                   f"Best layout: d={best_s['d']:.2f} m, "
                   f"theta={best_s['theta']:.2f} rad")

    # 报告
    def _clean(d_):
        return {k: v for k, v in d_.items() if not isinstance(v, np.ndarray)}
    report = {
        "best": _clean(best_s),
        "top_k": [_clean(s) for s, _ in refined],
        "cross_validation": agree,
    }
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=1)

    lines = [
        "# Rizon4 双臂 base 布局优化报告\n",
        f"**最优布局：d = {best_s['d']:.2f} m（两臂间距），"
        f"theta = {best_s['theta']:.2f} rad（向内 roll）**\n",
        "## 最优布局指标（直接 IK 精评）\n",
        f"- 可达率: {best_s['reach_rate']:.1%}"
        f"（左 {best_s['reach_rate_left']:.1%} / 右 {best_s['reach_rate_right']:.1%}）",
        f"- 平均归一化可操作度: 左 {best_s['mean_w_left']:.3f} / "
        f"右 {best_s['mean_w_right']:.3f}",
        f"- 臂间自碰撞率(配对, <{s_cfg['d_self_safe']*100:.0f}cm): "
        f"{best_s['collision_rate']:.1%}",
        f"- 高条件数占比(κ>{r_cfg['kappa_max']}): "
        f"左 {best_s['kappa_bad_rate_left']:.1%} / 右 {best_s['kappa_bad_rate_right']:.1%}",
        f"- 平均桌面裕度: {best_s['mean_table_margin']:.3f} m",
        f"- 共享区双臂可达率: {best_s['shared_both_reach']:.1%}\n",
        "## 查表 vs IK 交叉验证（最优布局）\n",
        f"- 查表可达率 {agree['scan_reach_rate']:.1%} ≤ "
        f"IK 可达率 {agree['ik_reach_rate']:.1%}（{agree['note']}）\n",
        "## Top-k 布局对比\n",
        "| # | d (m) | θ (rad) | 粗扫分 | 精评分 | 可达率 | 碰撞率 |",
        "|---|-------|---------|--------|--------|--------|--------|",
    ]
    for i, (s, _) in enumerate(refined):
        lines.append(f"| {i+1} | {s['d']:.2f} | {s['theta']:.2f} | "
                     f"{s['score_scan']:.3f} | {s['score_refined']:.3f} | "
                     f"{s['reach_rate']:.1%} | {s['collision_rate']:.1%} |")
    lines += ["", "图表见 `figs/`：`scan_heatmaps.png`（主交付物）、"
              "`capmap_{left,right}_D.png`、`best_layout_3d.png`。"]
    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write("\n".join(lines))
    print(f"\n最优: d={best_s['d']:.2f} m, theta={best_s['theta']:.2f} rad; "
          f"报告已存 {out_dir}/report.md")


def main():
    ap = argparse.ArgumentParser(description="Rizon4 base 布局优化管线")
    ap.add_argument("cmd", choices=["build-map", "scan", "refine", "all", "task-aabb"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--hdf5-root", default=None, help="task-aabb 用：真实采集 hdf5 数据集根目录")
    ap.add_argument("--max-files", type=int, default=None, help="task-aabb 用：调试时只处理前 N 个文件")
    ap.add_argument("--margin", type=float, default=0.0, help="task-aabb 用：AABB 四周留的安全余量(m)")
    args = ap.parse_args()
    cfg_path = args.config or os.path.join(PKG_DIR, "config.yaml")
    cfg = load_cfg(args.config)

    if args.cmd == "task-aabb":
        if not args.hdf5_root:
            sys.exit("task-aabb 需要 --hdf5-root <数据集根目录>")
        cmd_task_aabb(cfg, cfg_path, args.hdf5_root, max_files=args.max_files, margin=args.margin)
        return

    if args.cmd in ("build-map", "all"):
        if args.cmd == "all" and all(
                os.path.exists(map_path(cfg, s)) for s in ("left", "right")):
            print("能力图已存在，跳过 build-map（如需重建请单独运行 build-map）")
        else:
            cmd_build_map(cfg)
    if args.cmd in ("scan", "all"):
        cmd_scan(cfg)
    if args.cmd in ("refine", "all"):
        cmd_refine(cfg)


if __name__ == "__main__":
    sys.exit(main())
