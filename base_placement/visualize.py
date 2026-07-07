#!/usr/bin/env python3
"""可视化：能力图切片、(d,θ) 评分热力图、最优布局 3D 诊断散点。

配色遵循 dataviz 规范：
- 量值（sequential）：单色蓝 ramp（浅→深）
- 类别：左臂 蓝 #2a78d6 / 右臂 橙 #eb6834（固定槽位）
- 状态（问题点）：红 #e34948，配形状(x)与图例，不靠颜色单独编码
"""

from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SEQ_BLUE = LinearSegmentedColormap.from_list("seq_blue", [
    "#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
    "#256abf", "#184f95", "#0d366b",
])
C_LEFT, C_RIGHT, C_BAD = "#2a78d6", "#eb6834", "#e34948"
INK, INK2 = "#0b0b0b", "#52514e"


def _style_ax(ax):
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.tick_params(colors=INK2, labelsize=8)
    for s in ax.spines.values():
        s.set_color("#d0cfc9")


def plot_capmap_slices(cmap, save_path: str):
    """能力图可达指数 D 的三张正交切片（过 D 最大的体素）。"""
    D = cmap.reach_index_D()
    idx = np.unravel_index(np.argmax(D), D.shape)
    o, vs = cmap.origin, cmap.voxel_size
    names = ["x", "y", "z"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    slicers = [(D[idx[0], :, :].T, (1, 2)), (D[:, idx[1], :].T, (0, 2)),
               (D[:, :, idx[2]].T, (0, 1))]
    for ax, (img, (a, b)) in zip(axes, slicers):
        ext = [o[a], o[a] + D.shape[a] * vs, o[b], o[b] + D.shape[b] * vs]
        im = ax.imshow(img, origin="lower", extent=ext, cmap=SEQ_BLUE,
                       vmin=0, vmax=max(D.max(), 1e-6), aspect="equal")
        fixed = 3 - a - b
        ax.set_xlabel(f"{names[a]} (m)", color=INK2, fontsize=9)
        ax.set_ylabel(f"{names[b]} (m)", color=INK2, fontsize=9)
        ax.set_title(f"slice @ {names[fixed]}="
                     f"{o[fixed] + (idx[fixed] + 0.5) * vs:.2f} m",
                     color=INK, fontsize=10)
        _style_ax(ax)
    cb = fig.colorbar(im, ax=axes, shrink=0.85, aspect=25, pad=0.02)
    cb.set_label("reachability index D", color=INK2, fontsize=9)
    fig.suptitle(f"Capability map ({cmap.side} arm, base frame) — "
                 f"D = fraction of reachable orientation bins",
                 color=INK, fontsize=11)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {save_path}")


def plot_scan_heatmaps(results: list[dict], d_values, theta_values,
                       save_path: str):
    """(d,θ) 总分 + 分项指标热力图，标最优点。"""
    metrics = [("score", "Score (total)"),
               ("reach_rate", "Reach rate"),
               ("mean_w", "Mean manipulability (norm.)"),
               ("collision_rate", "Arm-arm collision rate"),
               ("mean_table_margin", "Mean table margin (m)"),
               ("overlap_quality", "Overlap quality (shared)")]
    nd, nt = len(d_values), len(theta_values)
    grids = {k: np.full((nd, nt), np.nan) for k, _ in metrics}
    lut = {(round(r["d"], 4), round(r["theta"], 4)): r for r in results}
    for i, d in enumerate(d_values):
        for j, t in enumerate(theta_values):
            r = lut[(round(float(d), 4), round(float(t), 4))]
            for k, _ in metrics:
                grids[k][i, j] = r[k]
    best = max(results, key=lambda r: r["score"])

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    ext = [theta_values[0], theta_values[-1], d_values[0], d_values[-1]]
    for ax, (k, title) in zip(axes.ravel(), metrics):
        im = ax.imshow(grids[k], origin="lower", extent=ext, aspect="auto",
                       cmap=SEQ_BLUE)
        ax.plot(best["theta"], best["d"], marker="*", ms=16, mec="white",
                mfc=C_BAD, mew=1.2)
        ax.set_xlabel("theta / inward roll (rad)", color=INK2, fontsize=9)
        ax.set_ylabel("d / base spacing (m)", color=INK2, fontsize=9)
        ax.set_title(title, color=INK, fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.9, aspect=18)
        _style_ax(ax)
    fig.suptitle(f"Base layout scan — best: d={best['d']:.2f} m, "
                 f"theta={best['theta']:.2f} rad (star)", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {save_path}")


def plot_layout_3d(refine_out: dict, tasks, z_table: float, save_path: str,
                   title: str):
    """最优布局诊断：任务点按 w_norm 着色（蓝 ramp），不可达点红 x；
    画两臂 base 位置与桌面示意。"""
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111, projection="3d")

    pts_all = []
    for gname, targets in [("left", tasks.left), ("right", tasks.right),
                           ("shared_left", tasks.shared)]:
        r = refine_out["points"][gname]
        pos = targets[:, :3, 3]
        pts_all.append(pos)
        ok = r["reachable"]
        sc = ax.scatter(pos[ok, 0], pos[ok, 1], pos[ok, 2],
                        c=r["w_norm"][ok], cmap=SEQ_BLUE, vmin=0, vmax=1,
                        s=22, depthshade=False)
        if (~ok).any():
            ax.scatter(pos[~ok, 0], pos[~ok, 1], pos[~ok, 2],
                       marker="x", s=40, c=C_BAD, linewidths=1.6,
                       label="unreachable" if gname == "left" else None)
    cb = fig.colorbar(sc, ax=ax, shrink=0.55, aspect=22, pad=0.06)
    cb.set_label("manipulability (normalized)", color=INK2, fontsize=9)

    # 两臂 base 与桌面
    for side, color in [("left", C_LEFT), ("right", C_RIGHT)]:
        T = refine_out["T_W"][side]
        p, ax_dir = T[:3, 3], T[:3, :3] @ np.array([0, 0, 0.25])
        ax.scatter(*p, marker="s", s=70, c=color, label=f"{side} base")
        ax.quiver(*p, *ax_dir, color=color, linewidth=2, arrow_length_ratio=0.2)
    pts = np.vstack(pts_all)
    xr = [pts[:, 0].min() - 0.15, pts[:, 0].max() + 0.15]
    yr = [pts[:, 1].min() - 0.15, pts[:, 1].max() + 0.15]
    xx, yy = np.meshgrid(xr, yr)
    ax.plot_surface(xx, yy, np.full_like(xx, z_table), alpha=0.12,
                    color="#9a9891")

    ax.set_xlabel("x (m)", color=INK2)
    ax.set_ylabel("y (m)", color=INK2)
    ax.set_zlabel("z (m)", color=INK2)
    ax.set_title(title, color=INK, fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_box_aspect([1, 1, 0.7])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {save_path}")
