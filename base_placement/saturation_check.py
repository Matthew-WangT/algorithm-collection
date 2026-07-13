#!/usr/bin/env python3
"""能力图采样饱和曲线：诊断 n_fk_samples 是否足够。

与其猜一个 n_fk_samples，不如实测：用递增的采样量 checkpoint（同一批
采样流累积，非重复独立建图）观察可达 bin 数、平均命中次数 avg_hits
（= 可达 bin 的采样命中次数均值）、singleton_frac（只被命中 1 次的
可达 bin 占比）随采样量增长是否已经拉平（类似生态学里的物种累积/
rarefaction 曲线）。

判读：
- avg_hits 越低、singleton_frac 越高，说明 best_w（取采样最大值）越
  不可信，reachable 也越可能有假阴性（真实可达但没采样命中）。
- 相邻 checkpoint 间"新增可达 bin"占比仍然很大，说明还在欠采样区间，
  继续加大 n_fk_samples 仍有明显收益；占比趋近 0 才算统计饱和。

用法：
  python -m base_placement.saturation_check --voxel-size 0.05 \
      --checkpoints 1e6 3e6 1e7 3e7
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time

import numpy as np

from .capability_map import CapabilityMap, _worker_chunk, _worker_init
from .robot_model import ArmModel, DEFAULT_URDF


def sampling_saturation_curve(
    side: str,
    urdf_path: str = DEFAULT_URDF,
    voxel_size: float = 0.05,
    n_dirs: int = 32,
    n_rolls: int = 8,
    n_workers: int = 16,
    chunk_size: int = 100_000,
    seed: int = 0,
    checkpoints: list[int] = (1_000_000, 3_000_000, 10_000_000, 30_000_000),
    margin_voxels: int = 2,
    verbose: bool = True,
) -> list[dict]:
    """单臂递增采样，在每个 checkpoint 记录可达 bin 数/平均命中次数/
    singleton 占比/w98，用于判断该 voxel_size 下 n_fk_samples 是否够。"""
    checkpoints = sorted(int(c) for c in checkpoints)
    arm = ArmModel(side, urdf_path)
    rng = np.random.default_rng(seed)

    n_pre = 50_000
    q_pre = arm.sample_arm_q(n_pre, rng)
    pos_pre = np.empty((n_pre, 3))
    for i in range(n_pre):
        pos_pre[i] = arm.fk_base(q_pre[i]).translation
    lo = pos_pre.min(axis=0) - margin_voxels * voxel_size
    hi = pos_pre.max(axis=0) + margin_voxels * voxel_size
    grid_shape = tuple(np.ceil((hi - lo) / voxel_size).astype(int))
    total_bins = int(np.prod(grid_shape)) * n_dirs * n_rolls
    if verbose:
        print(f"[{side}] grid_shape={grid_shape}, total_bins={total_bins:,}")

    cmap = CapabilityMap(lo, voxel_size, grid_shape, n_dirs, n_rolls, side)

    max_n = checkpoints[-1]
    n_chunks = int(np.ceil(max_n / chunk_size))
    tasks = [(seed + 1 + c, min(chunk_size, max_n - c * chunk_size))
             for c in range(n_chunks)]

    t0 = time.time()
    done = 0
    next_cp = 0
    rows = []
    with mp.Pool(n_workers, initializer=_worker_init,
                 initargs=(side, urdf_path)) as pool:
        for pos, R, w, q in pool.imap_unordered(_worker_chunk, tasks, chunksize=1):
            cmap.accumulate(pos, R, w, q)
            done += len(w)
            while next_cp < len(checkpoints) and done >= checkpoints[next_cp]:
                reachable_counts = cmap.count[cmap.count > 0]
                n_reach = int(reachable_counts.size)
                avg_hits = float(reachable_counts.mean()) if n_reach else 0.0
                singleton_frac = (float((reachable_counts == 1).mean())
                                   if n_reach else 0.0)
                cmap.finalize()
                row = dict(n_samples=checkpoints[next_cp], n_reach=n_reach,
                           pct=n_reach / total_bins, avg_hits=avg_hits,
                           singleton_frac=singleton_frac, w98=cmap.w98,
                           elapsed=time.time() - t0)
                rows.append(row)
                if verbose:
                    print(f"[{side}][{row['n_samples']:>11,}] "
                          f"reach={n_reach:>9,} ({row['pct']:.2%}) "
                          f"avg_hits={avg_hits:6.2f} "
                          f"singleton={singleton_frac:.1%} "
                          f"w98={cmap.w98:.4f} t={row['elapsed']:.0f}s")
                next_cp += 1
    return rows


def print_growth_report(rows: list[dict]):
    """相邻 checkpoint 间新增可达 bin 占比，用于判断曲线是否已拉平。"""
    print("\n--- 相邻档新增可达 bin 占比（判断是否饱和） ---")
    prev = 0
    for r in rows:
        delta = r["n_reach"] - prev
        growth_pct = delta / max(prev, 1) * 100
        print(f"  n={r['n_samples']:>11,}: 新增可达bin={delta:>9,} "
              f"(相对上一档 +{growth_pct:.1f}%)")
        prev = r["n_reach"]


def main():
    ap = argparse.ArgumentParser(
        description="能力图采样饱和曲线：诊断 n_fk_samples 是否足够")
    ap.add_argument("--side", default="left", choices=["left", "right"])
    ap.add_argument("--urdf-path", default=DEFAULT_URDF)
    ap.add_argument("--voxel-size", type=float, default=0.05)
    ap.add_argument("--n-dirs", type=int, default=32)
    ap.add_argument("--n-rolls", type=int, default=8)
    ap.add_argument("--n-workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoints", type=float, nargs="+",
                    default=[1e6, 3e6, 1e7, 3e7])
    args = ap.parse_args()

    rows = sampling_saturation_curve(
        args.side, args.urdf_path, args.voxel_size, args.n_dirs, args.n_rolls,
        args.n_workers, seed=args.seed,
        checkpoints=[int(c) for c in args.checkpoints])
    print_growth_report(rows)


if __name__ == "__main__":
    main()
