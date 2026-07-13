r"""递归提取目录下所有 hdf5 采集文件的双臂末端执行器（eef）工作空间，可视化并计算包围盒。

eef pose 在 hdf5 里以 world 系绝对 pose 存储，本脚本会尽量把它转换到机器人底座/骨盆参考系
下（`T_ref_eef = inv(T_w_ref) @ T_w_eef`），避免机器人整体移动（G1 走动、mocap 三角板移动）
污染"工作空间"范围。若某个文件缺少参考系数据，会使用未修正的 world 系坐标并在汇总里标注。

用法示例：
  python3 python/hdf5_eef_workspace.py \\
      --root-dir dataset/AssembleAirpodsV16/success \\
      --output-dir /tmp/workspace_test \\
      --max-files 20
"""

import argparse
import json
import os
import sys
import time

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

# active_mode -> (l_eef_link, r_eef_link, ref_link_for(robot_name))
# ref_link_for 为 None 表示该 active_mode 不提供参考系，直接使用未修正的 world 系。
LINK_TABLE = {
    "robot_vr": "cartesian",
    "robot_vla": "cartesian",
    "robot_vr_fixed_stand": "cartesian",
    "robot_vr_vla": "cartesian",
    "robot_sonic": "cartesian",
    "robot_locomanip": "cartesian",
    "record_handheld": "mocap_handheld",
    "record_mocap_umi": "mocap_umi",
    "record_mocap_whole_body": "mocap_umi",
}

UNSUPPORTED_MODES = {"roomtour3D"}


def resolve_link_names(active_mode, robot_name):
    """根据 active_mode/robot_model 解析左右 eef link 与参考 link 路径。

    返回 (l_eef_link, r_eef_link, ref_link) 或 None（不支持/无 eef 数据的 active_mode）。
    """
    if active_mode in UNSUPPORTED_MODES:
        return None

    kind = LINK_TABLE.get(active_mode)
    if kind is None:
        return None

    if kind == "cartesian":
        l_eef = "cartesian/left_gripper_f90c_end_effector_link"
        r_eef = "cartesian/right_gripper_f90c_end_effector_link"
        if robot_name == "unitree-g1":
            ref = "cartesian/pelvis"
        elif robot_name == "flexiv-rizon4":
            ref = "cartesian/pedestal"
        else:
            ref = None
        return l_eef, r_eef, ref

    if kind == "mocap_handheld":
        l_eef = "mocap/rigid_body/m_left_gripper_f90c_end_effector_link"
        r_eef = "mocap/rigid_body/m_right_gripper_f90c_end_effector_link"
        ref = "mocap/transformed/m_world"
        return l_eef, r_eef, ref

    if kind == "mocap_umi":
        l_eef = "mocap/rigid_body/m_left_gripper_f90c_end_effector_link"
        r_eef = "mocap/rigid_body/m_right_gripper_f90c_end_effector_link"
        ref = "cartesian/pelvis"
        return l_eef, r_eef, ref

    return None


def find_hdf5_files(root_dir):
    files = []
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".hdf5"):
                files.append(os.path.join(dirpath, filename))
    return sorted(files)


def read_pose_and_timestamp(f, link_path):
    """读取某个 link 的 pose(position+quaternion) 与对应 timestamp 数组。

    返回 (positions [T,3], quats [T,4] (xyzw), timestamps [T] 或 None) 或 None（数据集不存在）。
    """
    full_path = f"observations/{link_path}"
    if full_path not in f:
        return None
    dataset = f[full_path]

    if dataset.dtype.names is not None and "pose" in dataset.dtype.names:
        pose = dataset["pose"][:]
        pos = pose["position"]
        quat = pose["quaternion"]
    else:
        raise ValueError(f"未识别的 pose 数据集格式: {full_path}, dtype={dataset.dtype}")

    parent_path = f"observations/{os.path.dirname(link_path)}/timestamp"
    timestamps = f[parent_path][:] if parent_path in f else None

    return np.asarray(pos, dtype=np.float64), np.asarray(quat, dtype=np.float64), timestamps


def pose_to_mat(pos, quat):
    """Xyz + xyzw 四元数 -> [T,4,4] 齐次矩阵。"""
    mats = np.tile(np.eye(4), (pos.shape[0], 1, 1))
    mats[:, :3, 3] = pos
    mats[:, :3, :3] = Rotation.from_quat(quat).as_matrix()
    return mats


def align_to_timestamps(ref_pos, ref_quat, ref_ts, target_ts):
    """把参考 link 的 pose 最近邻对齐到目标（eef）的时间戳序列上。"""
    if ref_ts is None or target_ts is None:
        n = min(len(ref_pos), len(target_ts) if target_ts is not None else len(ref_pos))
        return ref_pos[:n], ref_quat[:n]

    idx = np.searchsorted(ref_ts, target_ts)
    idx = np.clip(idx, 1, len(ref_ts) - 1)
    left = idx - 1
    use_left = np.abs(ref_ts[left] - target_ts) <= np.abs(ref_ts[idx] - target_ts)
    idx = np.where(use_left, left, idx)
    return ref_pos[idx], ref_quat[idx]


def extract_file_points(path):
    """提取单个 hdf5 文件里左右 eef 在参考系下的位置点云。

    返回 dict: {"left": [N,3] or None, "right": [N,3] or None,
                "status": "ok"/"skipped"/"world_uncorrected", "reason": str}
    """
    with h5py.File(path, "r") as f:
        active_mode = f.attrs.get("active_mode")
        robot_name = f.attrs.get("robot_model", "unitree-g1")

        link_names = resolve_link_names(active_mode, robot_name)
        if link_names is None:
            return {"status": "skipped", "reason": f"不支持的 active_mode: {active_mode}"}
        l_eef_link, r_eef_link, ref_link = link_names

        result = {"status": "ok", "reason": "", "left": None, "right": None}

        ref_data = None
        if ref_link is not None:
            ref_data = read_pose_and_timestamp(f, ref_link)
            if ref_data is None:
                result["status"] = "world_uncorrected"
                result["reason"] = f"缺少参考系 {ref_link}，使用未修正 world 系坐标"

        for side, link in (("left", l_eef_link), ("right", r_eef_link)):
            eef_data = read_pose_and_timestamp(f, link)
            if eef_data is None:
                continue
            eef_pos, eef_quat, eef_ts = eef_data

            if ref_data is not None:
                ref_pos, ref_quat, ref_ts = ref_data
                aligned_ref_pos, aligned_ref_quat = align_to_timestamps(ref_pos, ref_quat, ref_ts, eef_ts)
                n = min(len(aligned_ref_pos), len(eef_pos))
                T_w_ref = pose_to_mat(aligned_ref_pos[:n], aligned_ref_quat[:n])
                T_w_eef = pose_to_mat(eef_pos[:n], eef_quat[:n])
                T_ref_eef = np.linalg.inv(T_w_ref) @ T_w_eef
                pos = T_ref_eef[:, :3, 3]
            else:
                pos = eef_pos

            mask = np.isfinite(pos).all(axis=-1)
            result[side] = pos[mask]

        if result["left"] is None and result["right"] is None:
            result["status"] = "skipped"
            result["reason"] = "该文件没有可用的 eef pose 数据"

        return result


def compute_aabb(points):
    if points is None or len(points) == 0:
        return None
    mn = points.min(axis=0)
    mx = points.max(axis=0)
    return {
        "min": mn.tolist(),
        "max": mx.tolist(),
        "size": (mx - mn).tolist(),
        "center": ((mn + mx) / 2).tolist(),
        "n_points": int(len(points)),
    }


def fmt_vec(v):
    return "(" + ", ".join(f"{x:.3f}" for x in v) + ")"


def print_summary(left_aabb, right_aabb, combined_aabb, n_files_ok, n_files_uncorrected, n_files_skipped):
    print()
    print("=" * 78)
    print("双臂 eef 工作空间汇总")
    print("=" * 78)
    print(f"处理文件: 正常修正={n_files_ok}, 世界系未修正={n_files_uncorrected}, 跳过={n_files_skipped}")
    for name, aabb in (("左臂", left_aabb), ("右臂", right_aabb), ("双臂合并", combined_aabb)):
        if aabb is None:
            print(f"{name}: 无有效数据")
            continue
        print(
            f"{name}: 点数={aabb['n_points']} "
            f"min={fmt_vec(aabb['min'])} max={fmt_vec(aabb['max'])} "
            f"size={fmt_vec(aabb['size'])} center={fmt_vec(aabb['center'])}"
        )


def plot_workspace(left_points, right_points, left_aabb, right_aabb, output_path, downsample):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for points, aabb, color, label in (
        (left_points, left_aabb, "tab:blue", "left"),
        (right_points, right_aabb, "tab:red", "right"),
    ):
        if points is None or len(points) == 0:
            continue
        sampled = points[::downsample]
        ax.scatter(sampled[:, 0], sampled[:, 1], sampled[:, 2], s=1, color=color, alpha=0.3, label=label)
        draw_bbox_wireframe(ax, aabb, color)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("dual-arm eef workspace (reference-frame relative)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def draw_bbox_wireframe(ax, aabb, color):
    if aabb is None:
        return
    mn = np.array(aabb["min"])
    mx = np.array(aabb["max"])
    corners = np.array([[x, y, z] for x in (mn[0], mx[0]) for y in (mn[1], mx[1]) for z in (mn[2], mx[2])])
    edges = [
        (0, 1),
        (0, 2),
        (0, 4),
        (1, 3),
        (1, 5),
        (2, 3),
        (2, 6),
        (3, 7),
        (4, 5),
        (4, 6),
        (5, 7),
        (6, 7),
    ]
    for i, j in edges:
        pts = corners[[i, j]]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, linewidth=0.8, alpha=0.6)


def scan_workspace(root_dir, max_files=None, arms="both", verbose=True) -> dict:
    """递归扫描 root_dir 下所有 hdf5 文件，提取双臂 eef 点云并计算 AABB。

    返回 dict: {"left_points", "right_points", "left_aabb", "right_aabb",
    "combined_aabb", "n_files_ok", "n_files_world_uncorrected",
    "n_files_skipped", "uncorrected_files", "skipped_files"}
    """
    root_dir = os.path.abspath(root_dir)
    if not os.path.isdir(root_dir):
        sys.exit(f"错误: 目录不存在: {root_dir}")

    files = find_hdf5_files(root_dir)
    if max_files is not None:
        files = files[:max_files]
    total = len(files)
    if total == 0:
        sys.exit(f"错误: {root_dir} 下没有找到 hdf5 文件")

    if verbose:
        print(f"共 {total} 个 hdf5 文件")

    left_points_list = []
    right_points_list = []
    n_ok, n_uncorrected, n_skipped = 0, 0, 0
    skipped_files = []
    uncorrected_files = []

    t_start = time.time()
    for idx, path in enumerate(files, start=1):
        rel_path = os.path.relpath(path, root_dir)
        try:
            result = extract_file_points(path)
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"[{idx}/{total}] {rel_path} 解析失败: {e}", file=sys.stderr)
            n_skipped += 1
            skipped_files.append({"file": rel_path, "reason": str(e)})
            continue

        if result["status"] == "skipped":
            n_skipped += 1
            skipped_files.append({"file": rel_path, "reason": result["reason"]})
        else:
            if result["status"] == "world_uncorrected":
                n_uncorrected += 1
                uncorrected_files.append({"file": rel_path, "reason": result["reason"]})
            else:
                n_ok += 1
            if arms in ("left", "both") and result["left"] is not None:
                left_points_list.append(result["left"])
            if arms in ("right", "both") and result["right"] is not None:
                right_points_list.append(result["right"])

        if verbose:
            elapsed = time.time() - t_start
            remaining = elapsed / idx * (total - idx)
            print(f"[{idx}/{total}] {rel_path} status={result['status']} 预计剩余={remaining / 60:.1f}min")

    left_points = np.concatenate(left_points_list, axis=0) if left_points_list else None
    right_points = np.concatenate(right_points_list, axis=0) if right_points_list else None
    combined_points = (
        np.concatenate([p for p in (left_points, right_points) if p is not None], axis=0)
        if left_points is not None or right_points is not None
        else None
    )

    return {
        "left_points": left_points,
        "right_points": right_points,
        "left_aabb": compute_aabb(left_points),
        "right_aabb": compute_aabb(right_points),
        "combined_aabb": compute_aabb(combined_points),
        "n_files_ok": n_ok,
        "n_files_world_uncorrected": n_uncorrected,
        "n_files_skipped": n_skipped,
        "uncorrected_files": uncorrected_files,
        "skipped_files": skipped_files,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root-dir", required=True, help="递归扫描的根目录")
    parser.add_argument(
        "--output-dir", default=None, help="输出 PNG + JSON 汇总的目录，默认 <root-dir>/workspace_analysis"
    )
    parser.add_argument("--downsample", type=int, default=5, help="可视化散点抽稀步长，包围盒统计不受影响")
    parser.add_argument("--max-files", type=int, default=None, help="调试用，只处理前 N 个文件")
    parser.add_argument("--arms", choices=["left", "right", "both"], default="both")
    args = parser.parse_args()

    root_dir = os.path.abspath(args.root_dir)
    output_dir = args.output_dir or os.path.join(root_dir, "workspace_analysis")
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {output_dir}")

    result = scan_workspace(root_dir, max_files=args.max_files, arms=args.arms, verbose=True)
    left_points = result["left_points"]
    right_points = result["right_points"]
    left_aabb = result["left_aabb"]
    right_aabb = result["right_aabb"]
    combined_aabb = result["combined_aabb"]
    n_ok = result["n_files_ok"]
    n_uncorrected = result["n_files_world_uncorrected"]
    n_skipped = result["n_files_skipped"]
    uncorrected_files = result["uncorrected_files"]
    skipped_files = result["skipped_files"]

    print_summary(left_aabb, right_aabb, combined_aabb, n_ok, n_uncorrected, n_skipped)
    if uncorrected_files:
        print("\n世界系未修正的文件:")
        for item in uncorrected_files:
            print(f"  - {item['file']}: {item['reason']}")
    if skipped_files:
        print("\n跳过的文件:")
        for item in skipped_files:
            print(f"  - {item['file']}: {item['reason']}")

    summary_path = os.path.join(output_dir, "workspace_summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                "n_files_ok": n_ok,
                "n_files_world_uncorrected": n_uncorrected,
                "n_files_skipped": n_skipped,
                "uncorrected_files": uncorrected_files,
                "skipped_files": skipped_files,
                "left_aabb": left_aabb,
                "right_aabb": right_aabb,
                "combined_aabb": combined_aabb,
            },
            f,
            indent=2,
        )
    print(f"\n汇总 JSON 已保存到: {summary_path}")

    if left_points is not None or right_points is not None:
        plot_path = os.path.join(output_dir, "workspace_bbox.png")
        plot_workspace(left_points, right_points, left_aabb, right_aabb, plot_path, args.downsample)
        print(f"可视化 PNG 已保存到: {plot_path}")


if __name__ == "__main__":
    main()
