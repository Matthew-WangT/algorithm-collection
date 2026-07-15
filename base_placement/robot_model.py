#!/usr/bin/env python3
"""Rizon4 双臂 URDF 的单臂子链封装与 base 布局位姿构造。

约定（与 serial_rizon4_with_gripper.urdf 及既有 sweep 脚本一致）：
- pedestal 根连杆位于世界原点，两臂经 pedestal_to_{left,right}_base 固定关节安装。
- 布局参数 (d, theta)：
    左臂  T_W_BL = Trans(x_base, +d/2, z_base) · RPY(-theta, pitch, 0)
    右臂  T_W_BR = Trans(x_base, -d/2, z_base) · RPY(+theta, pitch, 0)
  其中 pitch 默认 1.57（水平朝前安装），theta 为向内 roll 倾角。
  符号与 rizon4_workspace_sweep_y.py 的 modify_rizon4_base_y_r 一致。
- 能力图建在各臂 base_link 系下，T_B_E 与布局无关（这正是查表法成立的前提）。
"""

from __future__ import annotations

import os

import numpy as np
import pinocchio as pin

# 帧名 / 关节前缀模板（用 {side} 占位）——默认值对应 rizon4 双臂 URDF。
# 换机器人（如 s1/astribot）时不要改这里，改 config.yaml 的 robot 段即可，
# 由 arm_kwargs_from_cfg 覆盖。URDF 路径一律来自 config，代码不留硬编码路径。
DEFAULT_BASE_FRAME = "{side}_base_link"
DEFAULT_EE_FRAME = "{side}_gripper_f90c_end_effector_link"
DEFAULT_JOINT_PREFIX = "{side}_joint"

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))


def load_robot_cfg(config_path: str | None = None) -> dict:
    """读取 config.yaml 的 robot 段（供各模块 __main__ 冒烟共用）。"""
    import yaml
    with open(config_path or os.path.join(_PKG_DIR, "config.yaml")) as f:
        return yaml.safe_load(f)["robot"]


def arm_kwargs_from_cfg(robot_cfg: dict) -> dict:
    """把 config 的 robot 段转成 ArmModel 的关键字参数（缺字段回退 rizon4 模板）。"""
    return dict(
        urdf_path=robot_cfg["urdf_path"],
        base_frame=robot_cfg.get("base_frame", DEFAULT_BASE_FRAME),
        ee_frame=robot_cfg.get("ee_frame", DEFAULT_EE_FRAME),
        joint_prefix=robot_cfg.get("joint_prefix", DEFAULT_JOINT_PREFIX),
    )


def rpy_to_rotation(r: float, p: float, y: float) -> np.ndarray:
    """URDF 的 rpy 约定：R = Rz(y) · Ry(p) · Rx(r)。"""
    return (
        pin.utils.rotate("z", y) @ pin.utils.rotate("y", p) @ pin.utils.rotate("x", r)
    )


def make_base_pose(
    side: str,
    d: float,
    theta: float,
    x_base: float = 0.0,
    z_base: float = 1.4,
    pitch: float = 1.57,
) -> pin.SE3:
    """给定布局 (d, theta) 生成某臂 base 在世界系下的位姿 T_W_B。"""
    if side == "left":
        y, roll = +d / 2.0, -theta
    elif side == "right":
        y, roll = -d / 2.0, +theta
    else:
        raise ValueError(f"side 必须是 left/right: {side}")
    return pin.SE3(rpy_to_rotation(roll, pitch, 0.0), np.array([x_base, y, z_base]))


class ArmModel:
    """双臂 URDF 中单臂子链的采样 / FK / 雅可比封装。

    所有输出位姿均在该臂 base_link 系下（消掉 pedestal 安装，与布局无关）。
    """

    def __init__(self, side: str, urdf_path: str,
                 base_frame: str = DEFAULT_BASE_FRAME,
                 ee_frame: str = DEFAULT_EE_FRAME,
                 joint_prefix: str = DEFAULT_JOINT_PREFIX):
        if side not in ("left", "right"):
            raise ValueError(f"side 必须是 left/right: {side}")
        self.side = side
        self.urdf_path = os.path.abspath(urdf_path)
        self.model = pin.buildModelFromUrdf(self.urdf_path)
        self.data = self.model.createData()

        self.ee_frame_id = self._frame_id(ee_frame.format(side=side))
        self.base_frame_id = self._frame_id(base_frame.format(side=side))

        # 该臂的 7 个 revolute 关节（按 joint 名前缀识别，保持链序）
        prefix = joint_prefix.format(side=side)
        self.joint_ids = [
            jid
            for jid in range(1, self.model.njoints)
            if self.model.names[jid].startswith(prefix)
        ]
        if len(self.joint_ids) != 7:
            raise RuntimeError(
                f"{side} 臂关节数异常: {[self.model.names[j] for j in self.joint_ids]}"
            )
        self.q_indices = np.array([self.model.idx_qs[j] for j in self.joint_ids])
        self.v_indices = np.array([self.model.idx_vs[j] for j in self.joint_ids])
        self.lower = np.array(self.model.lowerPositionLimit)[self.q_indices]
        self.upper = np.array(self.model.upperPositionLimit)[self.q_indices]

        self.q_full_default = pin.neutral(self.model)

    def _frame_id(self, name: str) -> int:
        fid = self.model.getFrameId(name)
        if fid >= self.model.nframes:
            raise KeyError(f"未找到帧: {name}")
        return fid

    # ---------- 采样 ----------

    def sample_arm_q(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """在关节限位内均匀采样 n 组 7 维臂关节角。"""
        return rng.uniform(self.lower, self.upper, size=(n, 7))

    def full_q(self, q_arm: np.ndarray) -> np.ndarray:
        q = self.q_full_default.copy()
        q[self.q_indices] = q_arm
        return q

    # ---------- FK / 雅可比（均在 base_link 系） ----------

    def fk_base(self, q_arm: np.ndarray) -> pin.SE3:
        """EEF 在臂 base_link 系下的位姿 T_B_E。"""
        q = self.full_q(q_arm)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        oMb = self.data.oMf[self.base_frame_id]
        oMe = self.data.oMf[self.ee_frame_id]
        return oMb.actInv(oMe)

    def fk_and_manip(self, q_arm: np.ndarray) -> tuple[pin.SE3, float]:
        """T_B_E 及 Yoshikawa 可操作度 w = sqrt(det(J·Jᵀ))（6×7 帧雅可比）。

        w 对雅可比参考系的旋转不变，直接用 LOCAL 系计算。
        """
        q = self.full_q(q_arm)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        oMb = self.data.oMf[self.base_frame_id]
        oMe = self.data.oMf[self.ee_frame_id]
        J = pin.computeFrameJacobian(
            self.model, self.data, q, self.ee_frame_id, pin.ReferenceFrame.LOCAL
        )[:, self.v_indices]
        det = np.linalg.det(J @ J.T)
        w = float(np.sqrt(max(det, 0.0)))
        return oMb.actInv(oMe), w

    def joint_positions_base(self, q_arm: np.ndarray) -> np.ndarray:
        """该臂各关节原点 + EEF 原点在 base_link 系下的位置，(9, 3)。

        依次为 joint1..joint7 原点及 flange 前的 EEF 点，用作胶囊体端点链。
        """
        q = self.full_q(q_arm)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        oMb = self.data.oMf[self.base_frame_id]
        pts = [oMb.actInv(self.data.oMi[j]).translation for j in self.joint_ids]
        pts.append(oMb.actInv(self.data.oMf[self.ee_frame_id]).translation)
        # 补一个 base 原点作为链首（joint1 原点几乎与 base 重合，胶囊退化为点没关系）
        return np.array([np.zeros(3)] + pts)

    def jacobian_pos_cond(self, q_arm: np.ndarray) -> float:
        """位置块雅可比的条件数（LOCAL_WORLD_ALIGNED 位置 3 行）。"""
        q = self.full_q(q_arm)
        J = pin.computeFrameJacobian(
            self.model, self.data, q, self.ee_frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )[:3, self.v_indices]
        s = np.linalg.svd(J, compute_uv=False)
        return float(s[0] / max(s[-1], 1e-12))


def smoke_test() -> None:
    """步骤 1 自检：帧存在、T_B_E 与手工消装配一致、默认位形朝向合理。

    URDF 路径与帧命名从 config.yaml 的 robot 段读取（不留硬编码路径）。
    """
    robot_cfg = load_robot_cfg()
    akw = arm_kwargs_from_cfg(robot_cfg)
    for side in ("left", "right"):
        arm = ArmModel(side, **akw)
        rng = np.random.default_rng(0)
        q_arm = arm.sample_arm_q(1, rng)[0]

        # 一致性：T_B_E == inv(oMf_base) · oMf_ee（fk_base 内部即如此，交叉验证
        # 用 fk_and_manip 的另一条代码路径）
        T1 = arm.fk_base(q_arm)
        T2, w = arm.fk_and_manip(q_arm)
        assert np.allclose(T1.homogeneous, T2.homogeneous, atol=1e-10)
        assert w > 0

        # 零位形下 EEF 应在 base 前方（base 系 z 轴为臂伸出方向）
        T0 = arm.fk_base(np.zeros(7))
        print(f"[{side}] q=0 时 T_B_E 平移: {T0.translation.round(3)}, "
              f"EEF z 轴(base 系): {T0.rotation[:, 2].round(3)}, 随机位形 w={w:.4f}")

        # 关节位置链单调远离 base（粗查胶囊端点合理性）
        pts = arm.joint_positions_base(np.zeros(7))
        print(f"[{side}] 零位形关节链 |p|: {np.linalg.norm(pts, axis=1).round(3)}")

    # 布局位姿 vs URDF 原始安装：仅 rizon4（d=0.6,θ=0 → y=±0.3, rpy=(0,1.57,0)）
    # 时严格成立；换机器人（如 s1 臂装在 torso 上）安装位姿不同，仅作提示不断言。
    model = pin.buildModelFromUrdf(os.path.abspath(robot_cfg["urdf_path"]))
    data = model.createData()
    pin.framesForwardKinematics(model, data, pin.neutral(model))
    base_tmpl = robot_cfg.get("base_frame", DEFAULT_BASE_FRAME)
    for side in ("left", "right"):
        fid = model.getFrameId(base_tmpl.format(side=side))
        T_urdf = data.oMf[fid]
        T_ours = make_base_pose(side, d=0.6, theta=0.0)
        err_t = np.linalg.norm(T_urdf.translation - T_ours.translation)
        err_R = np.linalg.norm(T_urdf.rotation - T_ours.rotation)
        print(f"[{side}] make_base_pose vs URDF base: |Δt|={err_t:.2e}, "
              f"|ΔR|={err_R:.2e}（仅 rizon4 应为 0；其它机器人安装位姿本就不同）")
    print("smoke_test OK")


if __name__ == "__main__":
    smoke_test()
