#!/usr/bin/env python3
"""胶囊体碰撞几何：臂连杆的胶囊近似 + 胶囊-平面 / 胶囊-胶囊闭式距离。

胶囊链取自 ArmModel.joint_positions_base 的关节原点序列：
第 i 根胶囊 = 线段(p_i, p_{i+1}) + 半径 r_i。这是对连杆网格的粗包络，
半径按 Rizon4 连杆外径目测给定（可在 config 覆盖）。

桌面裕度约定：最后一段（flange→EEF，即 gripper）不参与桌面距离——
抓取本来就要接近桌面；量化的是肘/前臂等"不该蹭桌"的连杆。
臂间自碰撞则包含全部胶囊（gripper 互撞也是真风险）。
"""

from __future__ import annotations

import numpy as np

# 9 个链点 → 8 段胶囊的默认半径（m）：
# base→j1, j1→j2(肩), j2→j3(上臂), j3→j4(肘), j4→j5(前臂),
# j5→j6(腕1), j6→j7(腕2), j7→EEF(gripper)
DEFAULT_RADII = np.array([0.075, 0.075, 0.065, 0.065, 0.06, 0.055, 0.05, 0.055])
GRIPPER_SEGMENTS = 1  # 链末尾算作 gripper 的段数（桌面裕度时剔除）


def capsule_plane_distance(p0: np.ndarray, p1: np.ndarray, radii: np.ndarray,
                           z_plane: float) -> np.ndarray:
    """批量胶囊到水平面 z=z_plane 的有符号距离，(n,)。负值 = 穿透。

    p0, p1: (n,3) 各胶囊端点；radii: (n,)。
    """
    d = np.minimum(p0[:, 2], p1[:, 2]) - z_plane - radii
    return d


def segment_segment_distance(a0: np.ndarray, a1: np.ndarray,
                             b0: np.ndarray, b1: np.ndarray) -> np.ndarray:
    """批量线段-线段最短距离（向量化，含平行退化处理）。

    a0,a1,b0,b1: (n,3)。返回 (n,)。标准 clamped 最近点参数解法。
    """
    d1 = a1 - a0
    d2 = b1 - b0
    r = a0 - b0
    a = np.einsum("ij,ij->i", d1, d1)
    e = np.einsum("ij,ij->i", d2, d2)
    f = np.einsum("ij,ij->i", d2, r)
    c = np.einsum("ij,ij->i", d1, r)
    b = np.einsum("ij,ij->i", d1, d2)
    denom = a * e - b * b

    # 一般情形
    s = np.where(denom > 1e-12, (b * f - c * e) / np.where(denom > 1e-12, denom, 1.0), 0.0)
    s = np.clip(s, 0.0, 1.0)
    t = np.where(e > 1e-12, (b * s + f) / np.where(e > 1e-12, e, 1.0), 0.0)
    # t 出界时钳位并回代 s
    t_clamped = np.clip(t, 0.0, 1.0)
    need = t != t_clamped
    s = np.where(need,
                 np.clip((t_clamped * b - c) / np.where(a > 1e-12, a, 1.0), 0.0, 1.0),
                 s)
    t = t_clamped

    # 两线段都退化为点
    both_pt = (a <= 1e-12) & (e <= 1e-12)
    s = np.where(both_pt, 0.0, s)
    t = np.where(both_pt, 0.0, t)

    pa = a0 + s[:, None] * d1
    pb = b0 + t[:, None] * d2
    return np.linalg.norm(pa - pb, axis=1)


def capsule_capsule_min_distance(chain_a: np.ndarray, chain_b: np.ndarray,
                                 radii_a: np.ndarray = DEFAULT_RADII,
                                 radii_b: np.ndarray = DEFAULT_RADII) -> float:
    """两条胶囊链（(9,3) 链点）间的最小表面距离。负值 = 穿透。"""
    na, nb = len(chain_a) - 1, len(chain_b) - 1
    ia, ib = np.meshgrid(np.arange(na), np.arange(nb), indexing="ij")
    ia, ib = ia.ravel(), ib.ravel()
    d = segment_segment_distance(
        chain_a[ia], chain_a[ia + 1], chain_b[ib], chain_b[ib + 1]
    ) - radii_a[ia] - radii_b[ib]
    return float(d.min())


def chain_table_margin(chain: np.ndarray, z_table: float,
                       radii: np.ndarray = DEFAULT_RADII) -> float:
    """胶囊链（不含 gripper 段）到桌面的最小距离。"""
    n = len(chain) - 1 - GRIPPER_SEGMENTS
    d = capsule_plane_distance(chain[:n], chain[1:n + 1], radii[:n], z_table)
    return float(d.min())


def transform_chain(T: np.ndarray, chain: np.ndarray) -> np.ndarray:
    """(4,4) 齐次变换作用于 (m,3) 链点。"""
    return chain @ T[:3, :3].T + T[:3, 3]


def _self_test():
    # 平行线段
    d = segment_segment_distance(
        np.array([[0., 0, 0]]), np.array([[1., 0, 0]]),
        np.array([[0., 1, 0]]), np.array([[1., 1, 0]]))
    assert np.isclose(d[0], 1.0), d

    # 相交
    d = segment_segment_distance(
        np.array([[-1., 0, 0]]), np.array([[1., 0, 0]]),
        np.array([[0., -1, 0]]), np.array([[0., 1, 0]]))
    assert np.isclose(d[0], 0.0), d

    # 远离 + 端点最近
    d = segment_segment_distance(
        np.array([[0., 0, 0]]), np.array([[1., 0, 0]]),
        np.array([[3., 4, 0]]), np.array([[5., 4, 0]]))
    assert np.isclose(d[0], np.hypot(2, 4)), d

    # 退化为点
    d = segment_segment_distance(
        np.array([[0., 0, 0]]), np.array([[0., 0, 0]]),
        np.array([[0., 3, 0]]), np.array([[0., 3, 0]]))
    assert np.isclose(d[0], 3.0), d

    # 胶囊-平面
    dp = capsule_plane_distance(
        np.array([[0., 0, 0.5]]), np.array([[0., 0, 1.0]]),
        np.array([0.1]), 0.0)
    assert np.isclose(dp[0], 0.4), dp

    # 胶囊链-链
    chain_a = np.array([[0., 0, 0], [1., 0, 0]])
    chain_b = np.array([[0., 0.5, 0], [1., 0.5, 0]])
    dmin = capsule_capsule_min_distance(chain_a, chain_b,
                                        np.array([0.1]), np.array([0.1]))
    assert np.isclose(dmin, 0.3), dmin
    print("capsules self-test OK")


if __name__ == "__main__":
    _self_test()
