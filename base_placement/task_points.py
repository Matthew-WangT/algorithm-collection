#!/usr/bin/env python3
"""任务点集合：合成生成器（当前）+ UMI 轨迹 loader 接口（预留）。

世界系约定（与 URDF 一致）：pedestal 底座在原点，+x 为机器人前方，
+z 向上。桌面为 z = z_table 的水平面，任务区在 pedestal 前方桌面上空。

EEF 姿态约定（见 robot_model/capability_map）：**x 轴 = 抓取接近方向**，
z 轴 = 手指闭合方向。顶抓 = x 轴朝下 (0,0,-1)。

返回结构 TaskSet：
- left, right: (N,4,4) 各臂专属任务位姿（世界系）+ 权重
- shared: (M,4,4) 中央共享区位姿，左右臂都要评（overlap 指标用）
- 臂间碰撞检查用 left[i] 与 right[i] 配对（模拟双臂同时作业的典型组合）。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np


@dataclass
class TaskSet:
    left: np.ndarray            # (N,4,4)
    right: np.ndarray           # (N,4,4)
    shared: np.ndarray          # (M,4,4)
    w_left: np.ndarray = field(default=None)
    w_right: np.ndarray = field(default=None)
    w_shared: np.ndarray = field(default=None)

    def __post_init__(self):
        if self.w_left is None:
            self.w_left = np.ones(len(self.left))
        if self.w_right is None:
            self.w_right = np.ones(len(self.right))
        if self.w_shared is None:
            self.w_shared = np.ones(len(self.shared))

    def translated(self, p: np.ndarray) -> "TaskSet":
        """返回所有位姿刚性平移 p（3 维）后的新 TaskSet。

        用于"最佳操作区域中心 p 搜索"：局部系任务集只生成一次（固定 seed），
        不同候选 p 之间只平移不重采样，消除采样方差、保证候选可比。
        旋转与权重不变；返回副本，不原地修改。
        """
        p = np.asarray(p, dtype=float).reshape(3)

        def _shift(T: np.ndarray) -> np.ndarray:
            T2 = T.copy()
            T2[:, :3, 3] += p
            return T2

        return replace(
            self,
            left=_shift(self.left),
            right=_shift(self.right),
            shared=_shift(self.shared),
            w_left=self.w_left.copy(),
            w_right=self.w_right.copy(),
            w_shared=self.w_shared.copy(),
        )


def _pose(position: np.ndarray, approach: np.ndarray, closing_ref: np.ndarray
          ) -> np.ndarray:
    """由接近方向(EEF x)与手指闭合参考方向(EEF z)构造齐次位姿。

    closing_ref 会被投影到 approach 的垂面上并归一化。
    """
    x = approach / np.linalg.norm(approach)
    z = closing_ref - (closing_ref @ x) * x
    nz = np.linalg.norm(z)
    if nz < 1e-8:
        raise ValueError("closing_ref 与 approach 平行")
    z /= nz
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, position
    return T


def _sample_grasp_poses(n: int, x_range, y_range, z_range,
                        rng: np.random.Generator,
                        side_grasp_frac: float = 0.2,
                        max_tilt_deg: float = 30.0) -> np.ndarray:
    """在长方体区域内采样抓取位姿：顶抓（带随机倾斜）为主 + 少量侧抓。"""
    pos = np.stack([
        rng.uniform(*x_range, n),
        rng.uniform(*y_range, n),
        rng.uniform(*z_range, n),
    ], axis=1)
    poses = np.empty((n, 4, 4))
    n_side = int(round(n * side_grasp_frac))
    for i in range(n):
        if i < n_side:
            # 侧抓：接近方向水平（前向 ±45°），手指闭合方向水平
            phi = rng.uniform(-np.pi / 4, np.pi / 4)
            a = np.array([np.cos(phi), np.sin(phi), 0.0])
            closing = np.array([-np.sin(phi), np.cos(phi), 0.0])
        else:
            # 顶抓：接近方向从竖直向下随机倾斜 ≤ max_tilt
            tilt = np.deg2rad(rng.uniform(0.0, max_tilt_deg))
            az = rng.uniform(0.0, 2 * np.pi)
            a = np.array([
                np.sin(tilt) * np.cos(az),
                np.sin(tilt) * np.sin(az),
                -np.cos(tilt),
            ])
            # 闭合方向：水平面内随机 yaw
            psi = rng.uniform(0.0, 2 * np.pi)
            closing = np.array([np.cos(psi), np.sin(psi), 0.0])
        poses[i] = _pose(pos[i], a, closing)
    return poses


def synthetic_taskset(
    z_table: float = 0.75,
    x_range=(0.4, 0.9),
    y_left=(0.1, 0.4),
    y_right=(-0.4, -0.1),
    y_shared=(-0.15, 0.15),
    z_above=(0.05, 0.35),
    n_per_arm: int = 150,
    n_shared: int = 100,
    seed: int = 42,
) -> TaskSet:
    """按拷问敲定的默认场景生成合成任务点。"""
    rng = np.random.default_rng(seed)
    z_range = (z_table + z_above[0], z_table + z_above[1])
    left = _sample_grasp_poses(n_per_arm, x_range, y_left, z_range, rng)
    right = _sample_grasp_poses(n_per_arm, x_range, y_right, z_range, rng)
    shared = _sample_grasp_poses(n_shared, x_range, y_shared, z_range, rng)
    return TaskSet(left=left, right=right, shared=shared)


def centered_taskset(
    aabb_size=(0.5, 0.9),
    z_above=(0.05, 0.35),
    split_ratio=(0.4, 0.4),
    n_per_arm: int = 150,
    n_shared: int = 100,
    seed: int = 42,
) -> TaskSet:
    """在局部系生成任务点：原点为操作区 AABB 中心，z=0 平面为桌面。

    用于"最佳操作区域中心 p 搜索"模式：任务点只在局部系生成一次，
    配合 TaskSet.translated([px, 0, z_table + pz]) 平移到世界系候选中心，
    姿态采样逻辑与 synthetic_taskset 完全一致（同用 _sample_grasp_poses）。

    参数：
    - aabb_size = (Lx, Ly)：操作区 AABB 的 x/y 向尺寸 (m)；z 向高度带由
      z_above 给出（相对桌面，同 task.z_above 语义）。
    - split_ratio = (arm_frac, shared_frac)：沿 y 向三分区比例。左臂带占
      +y 侧 arm_frac*Ly，右臂带对称占 -y 侧，共享带为中央 shared_frac*Ly
      （允许与左右带部分重叠，保持现有三分区结构）。
    """
    Lx, Ly = float(aabb_size[0]), float(aabb_size[1])
    arm_frac, shared_frac = float(split_ratio[0]), float(split_ratio[1])
    rng = np.random.default_rng(seed)

    x_range = (-Lx / 2, Lx / 2)
    y_left = (Ly / 2 - arm_frac * Ly, Ly / 2)             # +y 侧带
    y_right = (-Ly / 2, -Ly / 2 + arm_frac * Ly)          # -y 侧带（对称）
    y_shared = (-shared_frac * Ly / 2, shared_frac * Ly / 2)
    z_range = (float(z_above[0]), float(z_above[1]))      # 桌面 z=0 基准

    left = _sample_grasp_poses(n_per_arm, x_range, y_left, z_range, rng)
    right = _sample_grasp_poses(n_per_arm, x_range, y_right, z_range, rng)
    shared = _sample_grasp_poses(n_shared, x_range, y_shared, z_range, rng)
    return TaskSet(left=left, right=right, shared=shared)


def load_umi_taskset(path: str, fmt: str = "npz") -> TaskSet:
    """UMI 轨迹 loader（预留）。

    TODO: 支持 npz/hdf5；相对位姿 → 绝对位姿展开（锚点对齐）；
    低速关键帧加权降采样。真实数据到位后实现，接口保持返回 TaskSet。
    """
    raise NotImplementedError("UMI 轨迹 loader 尚未实现，请先用 synthetic_taskset")


def _self_test():
    ts = synthetic_taskset()
    assert ts.left.shape == (150, 4, 4) and ts.shared.shape == (100, 4, 4)
    # 旋转矩阵正交、右手
    for T in [ts.left[0], ts.right[-1], ts.shared[3]]:
        R = T[:3, :3]
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-10)
    # 顶抓的接近方向（x 轴）应大体朝下
    n_down = (ts.left[:, 2, 0] < -0.85).sum()
    assert n_down >= 100, f"顶抓比例异常: {n_down}"
    # 位置都在任务区内
    assert ts.left[:, 1, 3].min() >= 0.1 and ts.right[:, 1, 3].max() <= -0.1

    # centered_taskset：局部系居中，x∈±Lx/2，左右臂 y 分居两侧，z 在高度带内
    cs = centered_taskset(aabb_size=(0.5, 0.9), z_above=(0.05, 0.35),
                          n_per_arm=20, n_shared=10, seed=1)
    assert cs.left.shape == (20, 4, 4) and cs.shared.shape == (10, 4, 4)
    eps = 1e-9
    assert abs(cs.left[:, 0, 3]).max() <= 0.25 + eps
    assert cs.left[:, 1, 3].min() > 0 and cs.right[:, 1, 3].max() < 0
    assert cs.left[:, 2, 3].min() >= 0.05 - eps
    assert cs.left[:, 2, 3].max() <= 0.35 + eps

    # translated：刚性平移，旋转/权重原样带过去，返回副本不改原对象
    p = np.array([0.9, 0.0, 0.8])
    orig = cs.left.copy()
    ct = cs.translated(p)
    assert np.allclose(ct.left[:, :3, 3], cs.left[:, :3, 3] + p)
    assert np.allclose(ct.right[:, :3, :3], cs.right[:, :3, :3])
    assert np.allclose(ct.w_shared, cs.w_shared) and ct.w_left is not cs.w_left
    assert np.allclose(cs.left, orig)  # 原对象未被修改
    print("task_points self-test OK")


if __name__ == "__main__":
    _self_test()
