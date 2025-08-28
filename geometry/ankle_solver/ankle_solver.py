import casadi as ca
import numpy as np

# 可视化配置选项
PLOT_CONFIG = {
    '3d_mode': 'scatter',  # 'scatter': 散点图 | 'surface': 3D表面
    'grid_resolution': 50,  # 表面网格分辨率 (建议: 30-100)
    'interpolation': 'cubic'  # 插值方法: 'linear'(线性) | 'nearest'(最近邻) | 'cubic'(三次样条)
}

# 使用说明:
# 可视化配置:
# 1. 修改 '3d_mode' 为 'scatter' 可显示原始数据点
# 2. 修改 '3d_mode' 为 'surface' 可显示插值表面
# 3. 'grid_resolution' 越高表面越平滑，但计算时间越长
# 4. 'interpolation' 方法影响表面平滑度: cubic > linear > nearest
#
# 数值计算:
# 雅可比矩阵求逆使用专用2x2解析公式，这是最快的方法

def to_float(casadi_val):
    """将CasADi类型转换为Python浮点数"""
    if isinstance(casadi_val, (ca.DM, ca.SX, ca.MX)):
        return float(casadi_val)
    return float(casadi_val)

def to_numpy(casadi_val):
    """将CasADi类型转换为numpy数组"""
    if isinstance(casadi_val, (ca.DM, ca.SX, ca.MX)):
        return np.array(casadi_val).flatten()
    return np.array(casadi_val).flatten()



def fast_2x2_inverse(A):
    """
    专门针对2x2矩阵的最快求逆方法
    使用解析公式: A^(-1) = (1/det) * [[d, -b], [-c, a]]
    """
    if A.shape != (2, 2):
        raise ValueError("This function is only for 2x2 matrices")
    
    a, b = A[0, 0], A[0, 1]
    c, d = A[1, 0], A[1, 1]
    
    # 计算行列式
    det = a * d - b * c
    
    # 检查奇异性
    if abs(det) < 1e-14:
        print("Warning: Matrix is nearly singular, using pseudo-inverse")
        return np.linalg.pinv(A)
    
    # 直接计算逆矩阵
    det_inv = 1.0 / det
    return np.array([[d * det_inv, -b * det_inv],
                     [-c * det_inv, a * det_inv]])



def euler_to_rotmat(roll, pitch, yaw):
    """
    欧拉角 (ZYX, yaw-pitch-roll) 转旋转矩阵
    输入: roll, pitch, yaw (casadi.SX 或 casadi.MX 或 float)
    输出: 3x3 旋转矩阵 (casadi.SX)
    """
    cr = ca.cos(roll)
    sr = ca.sin(roll)
    cp = ca.cos(pitch)
    sp = ca.sin(pitch)
    cy = ca.cos(yaw)
    sy = ca.sin(yaw)

    R = ca.vertcat(
        ca.horzcat(cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr),
        ca.horzcat(sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr),
        ca.horzcat(-sp,   cp*sr,             cp*cr)
    )
    return R


class AnkleInfo:
    D = 0.12
    d = D / 2.0
    h1 = 0.49
    h2 = 0.37
    r = 0.12

    delta_z = 0.00  # TODO: 跟十字轴的z距离

    p_lu_3 = ca.vertcat(-r, +d, 0.0)
    p_ru_3 = ca.vertcat(-r, -d, 0.0)

    p_la_1 = ca.vertcat(0, +d, h1)
    p_ra_1 = ca.vertcat(0, -d, h2)
    p_lb_1 = ca.vertcat(-r, +d, h1)
    p_rb_1 = ca.vertcat(-r, -d, h2)


class Ankle:
    def __init__(self, info: AnkleInfo):
        self.info = info

    def get_p_lu_1(self, pitch, roll):
        R = euler_to_rotmat(roll, pitch, 0)
        return R @ self.info.p_lu_3

    def get_p_ru_1(self, pitch, roll):
        R = euler_to_rotmat(roll, pitch, 0)
        return R @ self.info.p_ru_3

    def inv(self, pitch, roll, return_float=False):
        """
        逆运动学计算
        @param pitch: 俯仰角
        @param roll: 滚转角  
        @param return_float: 是否返回Python浮点数而不是CasADi类型
        @return: (phi_l, phi_r) 左右电机角度
        """
        p_lu_1 = self.get_p_lu_1(pitch, roll)
        p_ru_1 = self.get_p_ru_1(pitch, roll)
        phi_l = self._get_phi(p_lu_1, self.info.p_la_1, self.info.h1, self.info.r, return_float)
        phi_r = self._get_phi(p_ru_1, self.info.p_ra_1, self.info.h2, self.info.r, return_float)
        if return_float:
            return to_float(phi_l), to_float(phi_r)
        return phi_l, phi_r

    def forward(self, q_m, q_j0):
        """
        给定phi_l和phi_r，基于雅可比迭代方法，求解pitch和roll
        """
        jac = self._jacobian_func()
        q_j = q_j0
        phi_l0, phi_r0 = self.inv(q_j[0], q_j[1], return_float=True)
        q_m0 = np.array([phi_l0, phi_r0])
        err = np.linalg.norm(q_m - q_m0)
        iter_count = 0
        alpha = 0.9
        while err > 1e-4 and iter_count < 100:
            J = np.array(jac(q_j[0], q_j[1]))  # 转换为numpy数组~
            # 使用最快的2x2矩阵求逆方法
            J_inv = fast_2x2_inverse(J)
            delta_q_m = q_m - q_m0
            q_j = q_j + alpha * J_inv @ delta_q_m
            err = np.linalg.norm(delta_q_m)
            phi_l_new, phi_r_new = self.inv(q_j[0], q_j[1], return_float=True)
            q_m0 = np.array([phi_l_new, phi_r_new])
            iter_count += 1
        # print(f'iter_count: {iter_count}, err: {err}')
        return q_j, iter_count, err

    @staticmethod
    def _get_phi(p_u, p_a, h, r, return_float=False):
        d_ly = p_a[1] - p_u[1]
        l_xz = ca.sqrt(h**2 - d_ly**2)

        delta_x = p_a[0] - p_u[0]
        delta_z = p_a[2] - p_u[2]
        delta_l = ca.sqrt(delta_x**2 + delta_z**2)

        alpha = ca.arctan2(delta_x, delta_z)
        val = (delta_l**2 + r**2 - l_xz**2) / (2 * r * delta_l)
        val = ca.fmax(ca.fmin(val, 1.0), -1.0)  # 限制val在[-1, 1]范围内
        beta = ca.acos(val)
        phi = alpha + beta - ca.pi / 2

        return phi

    def _jacobian_func(self):
        roll = ca.SX.sym('roll')
        pitch = ca.SX.sym('pitch')
        phi_l, phi_r = self.inv(pitch, roll)
        jac = ca.jacobian(ca.vertcat(phi_l, phi_r), ca.vertcat(pitch, roll))
        func = ca.Function('jacobian', [pitch, roll], [jac])
        return func

    def jacobian(self, roll, pitch, return_numpy=False):
        """
        计算雅可比矩阵
        @param roll: 滚转角
        @param pitch: 俯仰角
        @param return_numpy: 是否返回numpy数组而不是CasADi类型
        @return: 2x2雅可比矩阵
        """
        func = self._jacobian_func()
        jac = func(roll, pitch)
        
        if return_numpy:
            return np.array(jac)
        return jac


def __main__():
    info = AnkleInfo()
    ankle = Ankle(info)

    print("p_lu_1 @ (0,0):", to_numpy(ankle.get_p_lu_1(0, 0)))
    print("p_ru_1 @ (0,0):", to_numpy(ankle.get_p_ru_1(0, 0)))
    print('='*30)

    # 测试CasADi类型输出
    phi_l0, phi_r0 = ankle.inv(0, 0)
    print(f'phi_l0 (CasADi): {phi_l0}, phi_r0 (CasADi): {phi_r0}')
    print(f'phi_l0.type: {type(phi_l0)}, phi_r0.type: {type(phi_r0)}')
    
    # 测试浮点数输出
    phi_l0_float, phi_r0_float = ankle.inv(0, 0, return_float=True)
    print(f'phi_l0 (float): {phi_l0_float}, phi_r0 (float): {phi_r0_float}')
    print(f'phi_l0_float.type: {type(phi_l0_float)}, phi_r0_float.type: {type(phi_r0_float)}')

    print("phi_l(-0.2, 0) (float):", ankle.inv(-0.2, 0, return_float=True))
    print("phi_l(0, -0.2) (float):", ankle.inv(0.0, -0.2, return_float=True))
    
    # 测试jacobian输出
    jac_casadi = ankle.jacobian(0.1, 0.2)
    jac_numpy = ankle.jacobian(0.1, 0.2, return_numpy=True)
    print("jacobian (CasADi):", jac_casadi)
    print("jacobian (numpy):", jac_numpy)
    print(f'jacobian types: CasADi={type(jac_casadi)}, numpy={type(jac_numpy)}')
    print('='*30)
    q_j, _, _ = ankle.forward(np.array([-0.2, -0.2]), np.array([0.0, 0.0]))
    print(f'q_j: {q_j}')
    print('='*30)
    solve_info = []
    solve_info_fail = []
    last_q_j = np.array([0.0,0.0])
    idx = 0
    use_last_q_j = False # 是否使用上一次的q_j作为初始值，但是此处实验的采样方法存在跳变，所以使用这个方法会导致部分点无法收敛
    for pitch in range(-80, 80, 3):
        pitch = 0.01 * pitch
        for roll in range(-80, 80, 3):
            roll = 0.01 * roll
            q_j_real = np.array([pitch, roll])
            phi_l, phi_r = ankle.inv(pitch, roll, return_float=True)
            if np.isnan(phi_l) or np.isnan(phi_r):
                print(f'idx[{idx}] exist nan: phi_l: {phi_l}, phi_r: {phi_r}')
                continue
            q_m = np.array([phi_l, phi_r])
            q_j, iter_count, err = ankle.forward(q_m, last_q_j)
            if err < 1e-3:
                solve_info.append([pitch, roll, iter_count, err])
            else:
                print(f'idx[{idx}]: pitch: {pitch}, roll: {roll}, phi_l: {phi_l}, phi_r: {phi_r}')
                solve_info_fail.append([pitch, roll, iter_count, err])
            if use_last_q_j:
                last_q_j = q_j
            idx += 1
            # print(f'pitch: {pitch}, roll: {roll}, phi_l: {phi_l}, phi_r: {phi_r}')
    # solve_info转换成numpy数组
    solve_info = np.array(solve_info)
    solve_info_fail = np.array(solve_info_fail)
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    if(len(solve_info) > 0):
        # 提取数据
        pitch_vals = solve_info[:, 0]
        roll_vals = solve_info[:, 1] 
        error_vals = solve_info[:, 3]
        
        # 3D图 - 成功的情况
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        if PLOT_CONFIG['3d_mode'] == "scatter":
            # 散点图模式
            scatter = ax.scatter(pitch_vals, roll_vals, error_vals, c=error_vals, cmap='viridis', alpha=0.7, s=20)
            ax.set_title('Forward Kinematics Error Points (Successful Cases)')
            plt.colorbar(scatter, shrink=0.8)
            
        elif PLOT_CONFIG['3d_mode'] == "surface":
            # 3D表面模式
            from scipy.interpolate import griddata
            
            # 定义网格范围
            pitch_min, pitch_max = pitch_vals.min(), pitch_vals.max()
            roll_min, roll_max = roll_vals.min(), roll_vals.max()
            
            # 创建网格
            grid_res = PLOT_CONFIG['grid_resolution']
            pitch_grid = np.linspace(pitch_min, pitch_max, grid_res)
            roll_grid = np.linspace(roll_min, roll_max, grid_res)
            PITCH, ROLL = np.meshgrid(pitch_grid, roll_grid)
            
            # 插值到网格上
            interp_method = PLOT_CONFIG['interpolation']
            ERROR = griddata((pitch_vals, roll_vals), error_vals, (PITCH, ROLL), 
                           method=interp_method, fill_value=np.nan)
            
            # 创建3D表面
            surface = ax.plot_surface(PITCH, ROLL, ERROR, cmap='viridis', alpha=0.8, edgecolor='none')
            ax.set_title(f'Forward Kinematics Error Surface ({interp_method} interpolation)')
            plt.colorbar(surface, shrink=0.8)
        
        ax.set_xlabel('Pitch (rad)')
        ax.set_ylabel('Roll (rad)')
        ax.set_zlabel('Error')
        plt.show()
        
        # 2D热图视图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 误差热图
        sc1 = ax1.scatter(pitch_vals, roll_vals, c=error_vals, cmap='viridis', alpha=0.7)
        ax1.set_xlabel('Pitch (rad)')
        ax1.set_ylabel('Roll (rad)')
        ax1.set_title('Error Distribution (Successful Cases)')
        ax1.grid(True)
        plt.colorbar(sc1, ax=ax1, label='Error')
        
        # 迭代次数热图
        iter_vals = solve_info[:, 2]
        sc2 = ax2.scatter(pitch_vals, roll_vals, c=iter_vals, cmap='plasma', alpha=0.7)
        ax2.set_xlabel('Pitch (rad)')
        ax2.set_ylabel('Roll (rad)')
        ax2.set_title('Iteration Count Distribution')
        ax2.grid(True)
        plt.colorbar(sc2, ax=ax2, label='Iterations')
        
        plt.tight_layout()
        plt.show()
    
    if(len(solve_info_fail) > 0):
        # 失败案例的可视化
        fig = plt.figure(figsize=(10, 6))
        
        pitch_fail = solve_info_fail[:, 0]
        roll_fail = solve_info_fail[:, 1]
        error_fail = solve_info_fail[:, 3]
        
        plt.scatter(pitch_fail, roll_fail, c=error_fail, cmap='Reds', alpha=0.8, s=50)
        plt.xlabel('Pitch (rad)')
        plt.ylabel('Roll (rad)')
        plt.title('Failed Cases Distribution')
        plt.colorbar(label='Error')
        plt.grid(True)
        plt.show()
    
    print(f"success: {len(solve_info)}, fail: {len(solve_info_fail)}")



if __name__ == "__main__":
    __main__()
