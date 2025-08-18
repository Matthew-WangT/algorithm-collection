import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def euler2matrix(zyx):
    assert len(zyx) == 3
    # 转换为弧度
    z_rad = np.radians(zyx[0])
    y_rad = np.radians(zyx[1])
    x_rad = np.radians(zyx[2])

    # 计算三角函数值
    cos_z, sin_z = np.cos(z_rad), np.sin(z_rad)
    cos_y, sin_y = np.cos(y_rad), np.sin(y_rad)
    cos_x, sin_x = np.cos(x_rad), np.sin(x_rad)

    Rz = np.array([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]])
    Ry = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])
    Rx = np.array([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]])

    # 组合旋转矩阵: R = Rx * Ry * Rz (左手系)
    R = Rx @ Ry @ Rz
    return R


def matrix2AxisAngle(R):
    # 计算旋转角度
    trace = np.trace(R)
    angle = np.arccos((trace - 1) / 2)

    # 如果角度接近0，返回零轴和零角度
    if np.abs(angle) < 1e-6:
        return np.array([0, 0, 1]), 0

    # 如果角度接近π，需要特殊处理
    if np.abs(angle - np.pi) < 1e-6:
        # 找到最大的对角元素
        diag = np.diag(R)
        max_idx = np.argmax(diag)

        if max_idx == 0:
            axis = np.array(
                [
                    np.sqrt((R[0, 0] + 1) / 2),
                    R[0, 1] / (2 * np.sqrt((R[0, 0] + 1) / 2)),
                    R[0, 2] / (2 * np.sqrt((R[0, 0] + 1) / 2)),
                ]
            )
        elif max_idx == 1:
            axis = np.array(
                [
                    R[1, 0] / (2 * np.sqrt((R[1, 1] + 1) / 2)),
                    np.sqrt((R[1, 1] + 1) / 2),
                    R[1, 2] / (2 * np.sqrt((R[1, 1] + 1) / 2)),
                ]
            )
        else:
            axis = np.array(
                [
                    R[2, 0] / (2 * np.sqrt((R[2, 2] + 1) / 2)),
                    R[2, 1] / (2 * np.sqrt((R[2, 2] + 1) / 2)),
                    np.sqrt((R[2, 2] + 1) / 2),
                ]
            )
    else:
        # 一般情况
        axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        axis = axis / (2 * np.sin(angle))

    # 归一化轴向量
    axis = axis / np.linalg.norm(axis)

    return axis, np.degrees(angle)


def hand_trans(R):
    T = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
    return T @ R @ T


def visualize_axis_angle(axis_l, angle_l, axis_r, angle_r):
    """可视化两个轴角表示"""
    fig = plt.figure(figsize=(15, 6))

    # 左手坐标系
    ax1 = fig.add_subplot(121, projection="3d")

    # 绘制坐标轴
    ax1.quiver(0, 0, 0, 1, 0, 0, color="red", arrow_length_ratio=0.1, label="X")
    ax1.quiver(0, 0, 0, 0, 1, 0, color="green", arrow_length_ratio=0.1, label="Y")
    ax1.quiver(0, 0, 0, 0, 0, 1, color="blue", arrow_length_ratio=0.1, label="Z")

    # 绘制旋转轴
    ax1.quiver(
        0,
        0,
        0,
        axis_l[0],
        axis_l[1],
        axis_l[2],
        color="purple",
        arrow_length_ratio=0.1,
        linewidth=3,
        label=f"Rotation Axis ({angle_l:.1f}°)",
    )

    # 绘制旋转圆弧示意
    t = np.linspace(0, np.radians(angle_l), 50)
    # 创建垂直于旋转轴的平面上的圆弧
    v1 = np.array([1, 0, 0]) if abs(axis_l[0]) < 0.9 else np.array([0, 1, 0])
    v1 = v1 - np.dot(v1, axis_l) * axis_l
    v1 = v1 / np.linalg.norm(v1) * 0.5
    v2 = np.cross(axis_l, v1)

    arc_points = np.array([np.cos(t)[:, None] * v1 + np.sin(t)[:, None] * v2]).squeeze()
    ax1.plot(arc_points[:, 0], arc_points[:, 1], arc_points[:, 2], "purple", alpha=0.7)

    ax1.set_xlim([-1, 1])
    ax1.set_ylim([-1, 1])
    ax1.set_zlim([-1, 1])
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_title(
        f"Left-Hand Coordinate System\nAxis: [{axis_l[0]:.3f}, {axis_l[1]:.3f}, {axis_l[2]:.3f}]\nAngle: {angle_l:.1f}°"
    )
    ax1.legend()

    # 右手坐标系
    ax2 = fig.add_subplot(122, projection="3d")

    # 绘制坐标轴
    ax2.quiver(0, 0, 0, 1, 0, 0, color="red", arrow_length_ratio=0.1, label="X")
    ax2.quiver(0, 0, 0, 0, 1, 0, color="green", arrow_length_ratio=0.1, label="Y")
    ax2.quiver(0, 0, 0, 0, 0, 1, color="blue", arrow_length_ratio=0.1, label="Z")

    # 绘制旋转轴
    ax2.quiver(
        0,
        0,
        0,
        axis_r[0],
        axis_r[1],
        axis_r[2],
        color="orange",
        arrow_length_ratio=0.1,
        linewidth=3,
        label=f"Rotation Axis ({angle_r:.1f}°)",
    )

    # 绘制旋转圆弧示意
    t = np.linspace(0, np.radians(angle_r), 50)
    v1 = np.array([1, 0, 0]) if abs(axis_r[0]) < 0.9 else np.array([0, 1, 0])
    v1 = v1 - np.dot(v1, axis_r) * axis_r
    v1 = v1 / np.linalg.norm(v1) * 0.5
    v2 = np.cross(axis_r, v1)

    arc_points = np.array([np.cos(t)[:, None] * v1 + np.sin(t)[:, None] * v2]).squeeze()
    ax2.plot(arc_points[:, 0], arc_points[:, 1], arc_points[:, 2], "orange", alpha=0.7)

    ax2.set_xlim([-1, 1])
    ax2.set_ylim([-1, 1])
    ax2.set_zlim([-1, 1])
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.set_title(
        f"Right-Hand Coordinate System\nAxis: [{axis_r[0]:.3f}, {axis_r[1]:.3f}, {axis_r[2]:.3f}]\nAngle: {angle_r:.1f}°"
    )
    ax2.legend()

    plt.tight_layout()
    plt.show()


def main():
    zyx = [60.0, 30, 20]
    R = euler2matrix(zyx)
    print(f"R: {R}")
    R_right = hand_trans(R)
    print(f"R_right: {R_right}")
    # R_left = hand_trans(R_right)
    # print(f"R_left: {R_left}")
    axis_l, angle_l = matrix2AxisAngle(R)
    axis_r, angle_r = matrix2AxisAngle(R_right)
    print(f"axis_l: {axis_l}, angle_l: {angle_l}")
    print(f"axis_r: {axis_r}, angle_r: {angle_r}")
    # 测试一个定义在左手系下的向量p_left=[0.1,0.2,0.3]在经历R旋转后的分别在左手系下的结果
    p_left = np.array([0.1, 0.2, 0.3])
    p_left2 = R @ p_left
    T = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
    p_right = T @ p_left
    p_right2 = R_right @ p_right
    print(f"p_left:  {p_left}")
    print(f"p_right: {p_right}")
    print(f"p_left2: {p_left2}")
    print(f"p_right2: {p_right2}")
    if(np.allclose(T @ p_left2, p_right2)):
        print("转换一致!!!")
    else:
        print("转换不一致!!!")

    # 可视化轴角表示
    # visualize_axis_angle(axis_l, angle_l, axis_r, angle_r)


if __name__ == "__main__":
    main()
