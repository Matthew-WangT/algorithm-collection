#!/usr/bin/env python3
"""单臂能力图（Zacharias capability map）：FK 大规模采样构建 + 姿态感知查询。

数据结构（全部在臂 base_link 系下）：
- 空间体素化：voxel_size（默认 5 cm），网格范围由预采样 AABB + 余量决定。
- 姿态 bin：Fibonacci 球 n_dirs 个接近方向 × n_rolls 档绕轴滚转。
  接近方向 = EEF frame 的 **x 轴**（F90C gripper 的伸出方向，见 URDF
  end_effector_joint 的 rpy 约定），不是惯用的 z 轴。
- 每 (体素, 姿态 bin) 存：FK 采样命中数、最佳 Yoshikawa 可操作度 w、
  对应的代表性关节角 q（供胶囊碰撞初筛与 IK warm-start）。

查询返回该 bin 的 {reachable, w_norm, q_repr}；w_norm 用全图可达 bin 的
98 分位归一化。体素级聚合可达指数 D = 可达姿态 bin 比例，用于经典可视化。
"""

from __future__ import annotations

import multiprocessing as mp
import time

import numpy as np

from .robot_model import ArmModel, DEFAULT_URDF


def fibonacci_sphere(n: int) -> np.ndarray:
    """球面近似均匀的 n 个单位方向，(n, 3)。"""
    i = np.arange(n, dtype=np.float64)
    phi = np.pi * (3.0 - np.sqrt(5.0)) * i
    z = 1.0 - 2.0 * (i + 0.5) / n
    r = np.sqrt(1.0 - z * z)
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def visualize_fibonacci_sphere(n: int = 256, save_path: str | None = None):
    """画出 fibonacci_sphere(n) 生成的方向点在单位球面上的分布，用于直观检查均匀性。

    先弹出交互窗口，可拖拽旋转查看球面分布；关闭窗口后（若给了 save_path）再落盘。
    """
    import matplotlib

    if matplotlib.get_backend().lower() == "qtagg":
        # 部分环境装了 Qt 但缺 libxcb-cursor0，qtagg 会直接崩溃；TkAgg 更可靠。
        try:
            matplotlib.use("TkAgg")
        except ImportError:
            pass
    import matplotlib.pyplot as plt

    dirs = fibonacci_sphere(n)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(dirs[:, 0], dirs[:, 1], dirs[:, 2], s=10, c=dirs[:, 2], cmap="viridis")
    ax.set_box_aspect((1, 1, 1))
    ax.set_title(f"fibonacci_sphere(n={n})")
    plt.show()  # 阻塞，用户可拖拽旋转；关闭窗口后再继续保存
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def tangent_bases(dirs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """为每个方向 d 构造正交切基 (t1, t2)，t1×t2 方向与 d 一致，用于定义 roll=0。"""
    ref = np.tile(np.array([0.0, 0.0, 1.0]), (len(dirs), 1))
    near_pole = np.abs(dirs[:, 2]) > 0.9
    ref[near_pole] = np.array([1.0, 0.0, 0.0])
    t1 = np.cross(ref, dirs)
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    t2 = np.cross(dirs, t1)
    return t1, t2


class CapabilityMap:
    def __init__(
        self,
        origin: np.ndarray,
        voxel_size: float,
        grid_shape: tuple[int, int, int],
        n_dirs: int,
        n_rolls: int,
        side: str,
    ):
        self.origin = np.asarray(origin, dtype=np.float64)
        self.voxel_size = float(voxel_size)
        self.grid_shape = tuple(grid_shape)
        self.n_dirs = int(n_dirs)
        self.n_rolls = int(n_rolls)
        self.side = side
        self.dirs = fibonacci_sphere(self.n_dirs)
        self.t1, self.t2 = tangent_bases(self.dirs)

        self.n_vox = int(np.prod(self.grid_shape))
        self.n_obins = self.n_dirs * self.n_rolls
        self.count = np.zeros(self.n_vox * self.n_obins, dtype=np.uint32)
        self.best_w = np.zeros(self.n_vox * self.n_obins, dtype=np.float32)
        self.best_q = np.zeros((self.n_vox * self.n_obins, 7), dtype=np.float16)
        self.w98 = 0.0

    # ---------- bin 量化（向量化，供构建与查询共用） ----------

    def voxel_index(self, pos: np.ndarray) -> np.ndarray:
        """(n,3) 位置 → (n,) 体素线性索引；出界返回 -1。"""
        ijk = np.floor((pos - self.origin) / self.voxel_size).astype(np.int64)
        ok = np.all((ijk >= 0) & (ijk < np.array(self.grid_shape)), axis=1)
        nx, ny, nz = self.grid_shape
        lin = (ijk[:, 0] * ny + ijk[:, 1]) * nz + ijk[:, 2]
        lin[~ok] = -1
        return lin

    def orient_index(self, R: np.ndarray) -> np.ndarray:
        """(n,3,3) 旋转矩阵 → (n,) 姿态 bin 索引。接近方向取 R[:, :, 0]（EEF x 轴）。"""
        approach = R[:, :, 0]
        d_idx = np.argmax(approach @ self.dirs.T, axis=1)
        # roll：EEF z 轴在该方向切平面上的方位角
        zaxis = R[:, :, 2]
        c = np.einsum("ij,ij->i", zaxis, self.t1[d_idx])
        s = np.einsum("ij,ij->i", zaxis, self.t2[d_idx])
        ang = np.arctan2(s, c)  # [-pi, pi)
        r_idx = np.floor((ang + np.pi) / (2 * np.pi) * self.n_rolls).astype(np.int64)
        r_idx = np.clip(r_idx, 0, self.n_rolls - 1)
        return d_idx * self.n_rolls + r_idx

    def bin_index(self, pos: np.ndarray, R: np.ndarray) -> np.ndarray:
        """(位置, 旋转) → 全局 bin 线性索引；位置出界返回 -1。"""
        v = self.voxel_index(pos)
        o = self.orient_index(R)
        lin = v * self.n_obins + o
        lin[v < 0] = -1
        return lin

    # ---------- 构建 ----------

    def accumulate(self, pos: np.ndarray, R: np.ndarray, w: np.ndarray, q: np.ndarray):
        ids = self.bin_index(pos, R)
        ok = ids >= 0
        ids, w, q = ids[ok], w[ok].astype(np.float32), q[ok]
        np.add.at(self.count, ids, 1)
        np.maximum.at(self.best_w, ids, w)
        new_best = w >= self.best_w[ids]
        self.best_q[ids[new_best]] = q[new_best].astype(np.float16)

    def finalize(self):
        reachable = self.best_w[self.count > 0]
        self.w98 = float(np.percentile(reachable, 98)) if reachable.size else 1.0

    # ---------- 查询 ----------

    def query(self, pos: np.ndarray, R: np.ndarray) -> dict:
        """批量查询 (n,3) 位置 + (n,3,3) 旋转（均在臂 base 系下）。"""
        ids = self.bin_index(pos, R)
        valid = ids >= 0
        ids_safe = np.where(valid, ids, 0)
        cnt = np.where(valid, self.count[ids_safe], 0)
        reachable = cnt > 0
        w_norm = np.clip(self.best_w[ids_safe] / max(self.w98, 1e-12), 0.0, 1.0)
        w_norm = np.where(reachable, w_norm, 0.0)
        q_repr = self.best_q[ids_safe].astype(np.float64)
        return {"reachable": reachable, "w_norm": w_norm, "q_repr": q_repr,
                "count": cnt}

    def reach_index_D(self) -> np.ndarray:
        """每体素聚合可达指数 D ∈ [0,1]，形状 grid_shape。"""
        hit = (self.count.reshape(self.n_vox, self.n_obins) > 0).mean(axis=1)
        return hit.reshape(self.grid_shape)

    def voxel_centers(self) -> np.ndarray:
        """所有体素中心坐标 (n_vox, 3)。"""
        nx, ny, nz = self.grid_shape
        ii, jj, kk = np.meshgrid(
            np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"
        )
        ijk = np.stack([ii, jj, kk], axis=-1).reshape(-1, 3)
        return self.origin + (ijk + 0.5) * self.voxel_size

    # ---------- 存取 ----------

    def save(self, path: str):
        np.savez_compressed(
            path,
            origin=self.origin, voxel_size=self.voxel_size,
            grid_shape=np.array(self.grid_shape), n_dirs=self.n_dirs,
            n_rolls=self.n_rolls, side=self.side,
            count=self.count, best_w=self.best_w, best_q=self.best_q,
            w98=self.w98,
        )

    @classmethod
    def load(cls, path: str) -> "CapabilityMap":
        z = np.load(path, allow_pickle=False)
        m = cls(z["origin"], float(z["voxel_size"]), tuple(z["grid_shape"]),
                int(z["n_dirs"]), int(z["n_rolls"]), str(z["side"]))
        m.count = z["count"]
        m.best_w = z["best_w"]
        m.best_q = z["best_q"]
        m.w98 = float(z["w98"])
        return m


# ---------- 并行构建 worker（模块级，便于 multiprocessing pickle） ----------

_WORKER_ARM: ArmModel | None = None


def _worker_init(side: str, urdf_path: str):
    global _WORKER_ARM
    _WORKER_ARM = ArmModel(side, urdf_path)


def _worker_chunk(args) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """采样 n 个位形，返回 (pos, R_flat, w, q)。"""
    seed, n = args
    arm = _WORKER_ARM
    rng = np.random.default_rng(seed)
    q_batch = arm.sample_arm_q(n, rng)
    pos = np.empty((n, 3))
    R = np.empty((n, 3, 3))
    w = np.empty(n)
    for i in range(n):
        T, wi = arm.fk_and_manip(q_batch[i])
        pos[i] = T.translation
        R[i] = T.rotation
        w[i] = wi
    return pos, R, w, q_batch


def build_capability_map(
    side: str,
    urdf_path: str = DEFAULT_URDF,
    n_samples: int = 10_000_000,
    voxel_size: float = 0.05,
    n_dirs: int = 32,
    n_rolls: int = 8,
    n_workers: int = 16,
    chunk_size: int = 100_000,
    seed: int = 0,
    margin_voxels: int = 2,
    verbose: bool = True,
) -> CapabilityMap:
    """FK 采样构建某臂能力图。先预采样定 AABB，再并行全量采样落 bin。"""
    t0 = time.time()
    arm = ArmModel(side, urdf_path)
    rng = np.random.default_rng(seed)

    # 预采样确定网格范围
    n_pre = 50_000
    q_pre = arm.sample_arm_q(n_pre, rng)
    pos_pre = np.empty((n_pre, 3))
    for i in range(n_pre):
        pos_pre[i] = arm.fk_base(q_pre[i]).translation
    lo = pos_pre.min(axis=0) - margin_voxels * voxel_size
    hi = pos_pre.max(axis=0) + margin_voxels * voxel_size
    grid_shape = tuple(np.ceil((hi - lo) / voxel_size).astype(int))
    if verbose:
        print(f"[{side}] AABB(base 系) {lo.round(2)} ~ {hi.round(2)}, "
              f"grid {grid_shape} = {np.prod(grid_shape):,} voxels, "
              f"bins/voxel {n_dirs * n_rolls}")

    cmap = CapabilityMap(lo, voxel_size, grid_shape, n_dirs, n_rolls, side)

    n_chunks = int(np.ceil(n_samples / chunk_size))
    tasks = [(seed + 1 + c, min(chunk_size, n_samples - c * chunk_size))
             for c in range(n_chunks)]
    with mp.Pool(n_workers, initializer=_worker_init, initargs=(side, urdf_path)) as pool:
        done = 0
        for pos, R, w, q in pool.imap_unordered(_worker_chunk, tasks, chunksize=1):
            cmap.accumulate(pos, R, w, q)
            done += len(w)
            if verbose and done % 1_000_000 < chunk_size:
                print(f"[{side}] {done:,}/{n_samples:,} samples, "
                      f"{time.time() - t0:.0f}s")
    cmap.finalize()
    if verbose:
        n_reach = int((cmap.count > 0).sum())
        print(f"[{side}] done in {time.time() - t0:.0f}s: "
              f"{n_reach:,} reachable bins "
              f"({n_reach / cmap.count.size:.1%} of {cmap.count.size:,}), "
              f"w98={cmap.w98:.4f}")
    return cmap


def self_check(cmap: CapabilityMap, arm: ArmModel, n: int = 200, seed: int = 123):
    """自洽断言：可达 bin 里存的 q 做 FK 应落回同一 bin。"""
    rng = np.random.default_rng(seed)
    reach_ids = np.flatnonzero(cmap.count > 0)
    ids = rng.choice(reach_ids, size=min(n, len(reach_ids)), replace=False)
    ok = 0
    for bid in ids:
        q = cmap.best_q[bid].astype(np.float64)
        T = arm.fk_base(q)
        rid = cmap.bin_index(T.translation[None], T.rotation[None])[0]
        ok += int(rid == bid)
    frac = ok / len(ids)
    print(f"self_check [{cmap.side}]: {ok}/{len(ids)} bins round-trip ({frac:.1%})")
    assert frac > 0.95, "bin 往返一致率过低，量化逻辑有 bug"


if __name__ == "__main__":
    # 小规模冒烟：1e5 样本建左臂图 + 自检
    m = build_capability_map("left", n_samples=100_000, n_workers=8)
    self_check(m, ArmModel("left"))
    # visualize_fibonacci_sphere(64, save_path='./fib_sphere_test.png')
