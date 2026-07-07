#!/usr/bin/env python3
"""用 MuJoCo 查看某个 (d, theta) 双臂 base 布局的 URDF。

原理：serial_rizon4_with_gripper.urdf 自带 <mujoco> compiler 标签，可直接被
MuJoCo 解析（无需转 MJCF）。本脚本复制一份 URDF，把
pedestal_to_{left,right}_base 两个 fixed joint 的 origin 替换成给定 (d, theta)
对应的位姿（与 robot_model.make_base_pose 的约定一致：
y=±d/2，roll=∓theta，pitch 固定）。

默认打开 MuJoCo 交互式 viewer（需要本机有显示，能用鼠标自由旋转/缩放查看）。
用的是 launch_passive，默认不跑动力学仿真（不会因为重力把手臂拽下垂），
只是把设定好的关节角摆出来给你看布局的静态几何关系。加 --out 则改为离屏渲染
保存一张静态 PNG（配合 MUJOCO_GL=egl 可在无显示环境下跑）。

用法：
    python render_layout.py --d 1.4 --theta 0.15          # 交互式查看
    MUJOCO_GL=egl python render_layout.py --d 1.4 --theta 0.15 \\
        --out /tmp/layout_opt.png                          # 存图，不弹窗

注意：pedestal（小车底座）网格尺寸不随 d 缩放，d 较大时右/左臂 base 会超出
底座边缘的可视化——这只是渲染没同步放大底座几何体，不代表算法计算有误；
实机部署前需按新的 base 间距重新设计底座。
"""

from __future__ import annotations

import argparse
import os
import re

import mujoco
import numpy as np

DEFAULT_URDF = (
    "/home/matthew/mindon/ik/rizon4_workspace_analysis/assets/rizon4/urdf/"
    "serial_rizon4_with_gripper.urdf"
)
Z_BASE = 1.4
PITCH = 1.57


def make_layout_urdf(src_path: str, d: float, theta: float, out_dir: str) -> str:
    """复制 URDF 并替换两臂 base 的安装 origin，返回新文件路径。"""
    with open(src_path) as f:
        text = f.read()

    # meshdir 是相对 URDF 文件位置的，复制到别处后需要改成绝对路径
    mesh_dir_abs = os.path.normpath(
        os.path.join(os.path.dirname(src_path), "..", "meshes")
    )
    text = text.replace('meshdir="../meshes"', f'meshdir="{mesh_dir_abs}"')

    y = d / 2.0
    left_origin = f'<origin rpy="{-theta} {PITCH} 0" xyz="0 {y} {Z_BASE}"/>'
    right_origin = f'<origin rpy="{theta} {PITCH} 0" xyz="0 {-y} {Z_BASE}"/>'

    text, n1 = re.subn(
        r'(<joint name="pedestal_to_left_base"[^>]*>\s*'
        r'<parent link="pedestal"/>\s*<child link="left_base_link"/>\s*)'
        r"<origin[^/]*/>",
        r"\1" + left_origin,
        text,
    )
    text, n2 = re.subn(
        r'(<joint name="pedestal_to_right_base"[^>]*>\s*'
        r'<parent link="pedestal"/>\s*<child link="right_base_link"/>\s*)'
        r"<origin[^/]*/>",
        r"\1" + right_origin,
        text,
    )
    if n1 != 1 or n2 != 1:
        raise RuntimeError(
            f"origin 替换失败（left 命中 {n1} 次，right 命中 {n2} 次），"
            "URDF 结构可能已变化，请检查正则"
        )

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"rizon4_d{d:.2f}_theta{theta:.2f}.urdf")
    with open(out_path, "w") as f:
        f.write(text)
    return out_path


def _elbow_bent_qpos(m: mujoco.MjModel) -> np.ndarray:
    """一个有辨识度的弯肘位形（14 dof: left_joint1..7, right_joint1..7）。"""
    q = np.zeros(m.nq)
    q[1], q[3], q[5] = -0.5, 1.3, 0.6        # left_joint2/4/6
    q[8], q[10], q[12] = -0.5, 1.3, 0.6      # right_joint2/4/6
    return q


def view_interactive(urdf_path: str) -> None:
    """打开 MuJoCo 交互式 viewer，自由旋转/缩放查看当前布局；不跑动力学仿真。

    用 launch_passive（而非 launch）：它没有后台物理线程，qpos 只在我们手动调用
    mj_step 时才会变化——这里从不调用，所以窗口默认就是“暂停”状态，不会因为
    重力把手臂拽垂下来，方便纯粹看 base 布局的静态几何关系。
    """
    import time

    import mujoco.viewer

    m = mujoco.MjModel.from_xml_path(urdf_path)
    d = mujoco.MjData(m)
    d.qpos[:] = _elbow_bent_qpos(m)
    mujoco.mj_forward(m, d)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            viewer.sync()
            time.sleep(1.0 / 60.0)


def render_to_png(
    urdf_path: str,
    out_png: str,
    title: str,
    width: int = 640,
    height: int = 480,
    azimuth: float = 130.0,
    elevation: float = -20.0,
    distance: float = 2.6,
    lookat=(0.3, 0.0, 1.1),
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = mujoco.MjModel.from_xml_path(urdf_path)
    d = mujoco.MjData(m)
    d.qpos[:] = _elbow_bent_qpos(m)
    mujoco.mj_forward(m, d)

    r = mujoco.Renderer(m, height=height, width=width)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    r.update_scene(d, camera=cam)
    img = r.render()

    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(title, fontsize=13)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_png}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--urdf", default=DEFAULT_URDF)
    ap.add_argument("--d", type=float, required=True, help="两臂 base 间距 (m)")
    ap.add_argument("--theta", type=float, required=True, help="向内 roll (rad)")
    ap.add_argument(
        "--out", default=None,
        help="给了就离屏渲染存成这个 PNG 路径；不给则打开交互式 viewer（默认）",
    )
    ap.add_argument("--title", default=None)
    ap.add_argument("--tmp-dir", default="/tmp/rizon4_layout_urdf")
    ap.add_argument("--azimuth", type=float, default=130.0)
    ap.add_argument("--elevation", type=float, default=-20.0)
    ap.add_argument("--distance", type=float, default=2.6)
    args = ap.parse_args()

    urdf = make_layout_urdf(args.urdf, args.d, args.theta, args.tmp_dir)

    if args.out is None:
        print(f"打开交互式 viewer: d={args.d:.2f} m, theta={args.theta:.2f} rad")
        view_interactive(urdf)
    else:
        title = args.title or f"d={args.d:.2f} m, theta={args.theta:.2f} rad"
        render_to_png(
            urdf, args.out, title,
            azimuth=args.azimuth, elevation=args.elevation,
            distance=args.distance,
        )


if __name__ == "__main__":
    main()
