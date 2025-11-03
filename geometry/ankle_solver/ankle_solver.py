import casadi as ca
import numpy as np
import os
import subprocess
import ctypes
from ctypes import POINTER, c_double, c_int, c_longlong

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
    d1 = 12.76
    d2 = 12.76
    h1 = 100
    h2 = 170
    r1 = 40
    r2 = 40
    u_x = 36
    u_z = 0.00 #跟十字轴的z距离

    p_lu_3 = ca.vertcat(u_x, +d1, u_z)
    p_ru_3 = ca.vertcat(u_x, -d2, u_z)

    p_la_1 = ca.vertcat(0, +d1, h1)
    p_ra_1 = ca.vertcat(0, -d2, h2)
    p_lb_1 = ca.vertcat(-r1, +d1, h1)
    p_rb_1 = ca.vertcat(-r2, -d2, h2)


class Ankle:
    def __init__(self, info: AnkleInfo):
        self.info = info

    def get_p_lu_1(self, pitch, roll):
        R = euler_to_rotmat(roll, pitch, 0)
        return R @ self.info.p_lu_3

    def get_p_ru_1(self, pitch, roll):
        R = euler_to_rotmat(roll, pitch, 0)
        return R @ self.info.p_ru_3
    
    def _get_p_u_1(self, pitch, roll, p_u_3):
        """计算旋转后的点坐标"""
        R = euler_to_rotmat(roll, pitch, 0)
        return R @ p_u_3

    def inv(self, pitch, roll, d1=None, d2=None, h1=None, h2=None, r1=None, r2=None, u_x=None, u_z=None, return_float=False):
        """
        逆运动学计算
        @param pitch: 俯仰角
        @param roll: 滚转角
        @param d1: 左侧十字轴间距的一半 (可选，默认使用info中的值)
        @param d2: 右侧十字轴间距的一半 (可选，默认使用info中的值)
        @param h1: 左侧连杆长度 (可选，默认使用info中的值)
        @param h2: 右侧连杆长度 (可选，默认使用info中的值)
        @param r1: 左侧连杆偏移距离 (可选，默认使用info中的值)
        @param r2: 右侧连杆偏移距离 (可选，默认使用info中的值)
        @param u_x: 十字轴x偏移 (可选，默认使用info中的值)
        @param u_z: 十字轴z偏移 (可选，默认使用info中的值)
        @param return_float: 是否返回Python浮点数而不是CasADi类型
        @return: (phi_l, phi_r) 左右电机角度
        """
        # 如果参数未提供，使用默认值
        if d1 is None: d1 = self.info.d1
        if d2 is None: d2 = self.info.d2
        if h1 is None: h1 = self.info.h1
        if h2 is None: h2 = self.info.h2
        if r1 is None: r1 = self.info.r1
        if r2 is None: r2 = self.info.r2
        if u_x is None: u_x = self.info.u_x
        if u_z is None: u_z = self.info.u_z
        
        p_lu_3 = ca.vertcat(u_x, +d1, u_z)
        p_ru_3 = ca.vertcat(u_x, -d2, u_z)
        p_la_1 = ca.vertcat(0, +d1, h1)
        p_ra_1 = ca.vertcat(0, -d2, h2)
        
        p_lu_1 = self._get_p_u_1(pitch, roll, p_lu_3)
        p_ru_1 = self._get_p_u_1(pitch, roll, p_ru_3)
        phi_l = self._get_phi(p_lu_1, p_la_1, h1, r1, return_float)
        phi_r = self._get_phi(p_ru_1, p_ra_1, h2, r2, return_float)
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

        delta_x = ca.fabs(p_a[0] - p_u[0])
        delta_z = p_a[2] - p_u[2]
        delta_l = ca.sqrt(delta_x**2 + delta_z**2)

        alpha = ca.arctan2(delta_x, delta_z)
        val = (delta_l**2 + r**2 - l_xz**2) / (2 * r * delta_l)
        val = ca.fmax(ca.fmin(val, 1.0), -1.0)  # 限制val在[-1, 1]范围内
        beta = ca.acos(val)
        phi = alpha + beta - ca.pi / 2

        return -phi * ca.sign(p_u[0])

    def _jacobian_func(self, with_params=False):
        roll = ca.SX.sym('roll')
        pitch = ca.SX.sym('pitch')
        
        if with_params:
            # 创建参数符号变量
            d1 = ca.SX.sym('d1')
            d2 = ca.SX.sym('d2')
            h1 = ca.SX.sym('h1')
            h2 = ca.SX.sym('h2')
            r1 = ca.SX.sym('r1')
            r2 = ca.SX.sym('r2')
            u_x = ca.SX.sym('u_x')
            u_z = ca.SX.sym('u_z')
            
            phi_l, phi_r = self.inv(pitch, roll, d1, d2, h1, h2, r1, r2, u_x, u_z)
            jac = ca.jacobian(ca.vertcat(phi_l, phi_r), ca.vertcat(pitch, roll))
            func = ca.Function('jacobian', [pitch, roll, d1, d2, h1, h2, r1, r2, u_x, u_z], [jac])
        else:
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

    def export_cpp(self, out_dir: str, prefix: str = "ankle", params_only: bool = True):
        """
        将 IK 与 Jacobian 导出为可由 C++ 编译使用的 C 源码与头文件。
        生成文件: {out_dir}/{prefix}_functions.c 与 {out_dir}/{prefix}_functions.h

        @param out_dir: 输出目录
        @param prefix: 函数名前缀
        @param params_only: 是否只导出带参数的函数版本（默认True）
        
        用法示例:
            ankle.export_cpp("./build", prefix="ankle")  # 只导出带参数版本
            ankle.export_cpp("./build", prefix="ankle", params_only=False)  # 导出传统版本
        """
        os.makedirs(out_dir, exist_ok=True)

        # 符号变量
        pitch = ca.SX.sym('pitch')
        roll = ca.SX.sym('roll')

        if params_only:
            # 创建参数符号变量
            d1 = ca.SX.sym('d1')
            d2 = ca.SX.sym('d2')
            h1 = ca.SX.sym('h1')
            h2 = ca.SX.sym('h2')
            r1 = ca.SX.sym('r1')
            r2 = ca.SX.sym('r2')
            u_x = ca.SX.sym('u_x')
            u_z = ca.SX.sym('u_z')
            
            # IK 与 Jacobian
            phi_l, phi_r = self.inv(pitch, roll, d1, d2, h1, h2, r1, r2, u_x, u_z)
            J = ca.jacobian(ca.vertcat(phi_l, phi_r), ca.vertcat(pitch, roll))

            f_inv = ca.Function(f"{prefix}_inv", 
                                [pitch, roll, d1, d2, h1, h2, r1, r2, u_x, u_z], 
                                [phi_l, phi_r],
                                ["pitch", "roll", "d1", "d2", "h1", "h2", "r1", "r2", "u_x", "u_z"], 
                                ["phi_l", "phi_r"])
            f_jac = ca.Function(f"{prefix}_jacobian", 
                                [pitch, roll, d1, d2, h1, h2, r1, r2, u_x, u_z], 
                                [J],
                                ["pitch", "roll", "d1", "d2", "h1", "h2", "r1", "r2", "u_x", "u_z"], 
                                ["J"])
        else:
            # IK 与 Jacobian（传统版本，使用硬编码参数）
            phi_l, phi_r = self.inv(pitch, roll)
            J = ca.jacobian(ca.vertcat(phi_l, phi_r), ca.vertcat(pitch, roll))

            f_inv = ca.Function(f"{prefix}_inv", [pitch, roll], [phi_l, phi_r],
                                ["pitch", "roll"], ["phi_l", "phi_r"])
            f_jac = ca.Function(f"{prefix}_jacobian", [pitch, roll], [J],
                                ["pitch", "roll"], ["J"])

        # 代码生成（CasADi 生成 C 代码，可被 C++ 直接编译链接）
        # 注意：CodeGenerator 的参数必须是一个"模块名"，不能包含路径或扩展名
        module_name = f"{prefix}_functions"
        cg = ca.CodeGenerator(module_name)
        cg.add(f_inv)
        cg.add(f_jac)
        # 在指定目录内生成文件
        cwd = os.getcwd()
        try:
            os.chdir(out_dir)
            cg.generate()
        finally:
            os.chdir(cwd)

        # 返回生成的 c 文件路径
        c_path = os.path.join(out_dir, f"{module_name}.c")
        return c_path

    def compile_shared_library(self, c_file_path: str, output_name: str = None, optimize: bool = True):
        """
        将生成的C文件编译为共享库(.so文件)，以便在Python中调用
        
        @param c_file_path: C文件路径
        @param output_name: 输出库文件名（不含扩展名），默认与C文件名相同
        @param optimize: 是否启用优化编译（默认True）
        @return: 编译后的共享库路径
        """
        if not os.path.exists(c_file_path):
            raise FileNotFoundError(f"C文件不存在: {c_file_path}")
        
        out_dir = os.path.dirname(c_file_path)
        if output_name is None:
            output_name = os.path.splitext(os.path.basename(c_file_path))[0]
        
        # 确定共享库扩展名
        if os.name == 'nt':  # Windows
            lib_ext = '.dll'
        else:  # Linux/Mac
            lib_ext = '.so'
        
        lib_path = os.path.join(out_dir, f"{output_name}{lib_ext}")
        
        # 编译命令
        if optimize:
            compile_flags = ['-O3', '-shared', '-fPIC']
        else:
            compile_flags = ['-shared', '-fPIC']
        
        cmd = ['gcc'] + compile_flags + ['-o', lib_path, c_file_path, '-lm']
        
        print(f"正在编译共享库: {lib_path}")
        print(f"编译命令: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"编译成功: {lib_path}")
            return lib_path
        except subprocess.CalledProcessError as e:
            print(f"编译失败:")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            raise RuntimeError(f"编译失败: {e}")


class AnkleCWrapper:
    """
    Python包装器，用于调用编译后的C函数
    """
    def __init__(self, lib_path: str, info: AnkleInfo = None):
        """
        加载C共享库
        
        @param lib_path: 共享库文件路径
        @param info: AnkleInfo实例，用于管理几何参数（默认使用AnkleInfo的默认值）
        """
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"共享库不存在: {lib_path}")
        
        # 使用AnkleInfo管理参数
        if info is None:
            self.info = AnkleInfo()
        else:
            self.info = info
        
        self.lib = ctypes.CDLL(lib_path)
        
        # 定义函数签名
        # CasADi函数的签名: int func(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem)
        # casadi_real = double, casadi_int = long long int
        
        # ankle_inv函数
        self.lib.ankle_inv.argtypes = [
            POINTER(POINTER(c_double)),  # arg
            POINTER(POINTER(c_double)),  # res
            POINTER(c_longlong),          # iw
            POINTER(c_double),            # w
            c_int                         # mem
        ]
        self.lib.ankle_inv.restype = c_int
        
        # ankle_jacobian函数
        self.lib.ankle_jacobian.argtypes = [
            POINTER(POINTER(c_double)),  # arg
            POINTER(POINTER(c_double)),  # res
            POINTER(c_longlong),          # iw
            POINTER(c_double),            # w
            c_int                         # mem
        ]
        self.lib.ankle_jacobian.restype = c_int
        
        # 获取输入输出数量
        self.lib.ankle_inv_n_in.restype = c_longlong
        self.lib.ankle_inv_n_out.restype = c_longlong
        self.lib.ankle_jacobian_n_in.restype = c_longlong
        self.lib.ankle_jacobian_n_out.restype = c_longlong
        
        # 获取工作空间大小
        self.lib.ankle_inv_work.argtypes = [
            POINTER(c_longlong),  # sz_arg
            POINTER(c_longlong),  # sz_res
            POINTER(c_longlong),  # sz_iw
            POINTER(c_longlong)   # sz_w
        ]
        self.lib.ankle_inv_work.restype = c_int
        
        self.lib.ankle_jacobian_work.argtypes = [
            POINTER(c_longlong),  # sz_arg
            POINTER(c_longlong),  # sz_res
            POINTER(c_longlong),  # sz_iw
            POINTER(c_longlong)   # sz_w
        ]
        self.lib.ankle_jacobian_work.restype = c_int
        
        # 初始化工作空间
        sz_arg = c_longlong()
        sz_res = c_longlong()
        sz_iw = c_longlong()
        sz_w = c_longlong()
        
        self.lib.ankle_inv_work(ctypes.byref(sz_arg), ctypes.byref(sz_res), 
                                ctypes.byref(sz_iw), ctypes.byref(sz_w))
        self.inv_sz_arg = sz_arg.value
        self.inv_sz_res = sz_res.value
        self.inv_sz_iw = sz_iw.value
        self.inv_sz_w = sz_w.value
        
        self.lib.ankle_jacobian_work(ctypes.byref(sz_arg), ctypes.byref(sz_res), 
                                     ctypes.byref(sz_iw), ctypes.byref(sz_w))
        self.jac_sz_arg = sz_arg.value
        self.jac_sz_res = sz_res.value
        self.jac_sz_iw = sz_iw.value
        self.jac_sz_w = sz_w.value
    
    def inv(self, pitch, roll, d1=None, d2=None, h1=None, h2=None, 
            r1=None, r2=None, u_x=None, u_z=None):
        """
        调用C版本的逆运动学函数
        
        @param pitch: 俯仰角
        @param roll: 滚转角
        @param d1, d2, h1, h2, r1, r2, u_x, u_z: 几何参数（可选，默认使用AnkleInfo的值）
        @return: (phi_l, phi_r) 左右电机角度
        """
        # 如果参数未提供，使用AnkleInfo中的值
        if d1 is None: d1 = self.info.d1
        if d2 is None: d2 = self.info.d2
        if h1 is None: h1 = self.info.h1
        if h2 is None: h2 = self.info.h2
        if r1 is None: r1 = self.info.r1
        if r2 is None: r2 = self.info.r2
        if u_x is None: u_x = self.info.u_x
        if u_z is None: u_z = self.info.u_z
        
        # 准备输入数组
        inputs = [pitch, roll, d1, d2, h1, h2, r1, r2, u_x, u_z]
        arg_array = (POINTER(c_double) * self.inv_sz_arg)()
        for i, val in enumerate(inputs):
            arg_array[i] = (c_double * 1)(val)
        
        # 准备输出数组
        res_array = (POINTER(c_double) * self.inv_sz_res)()
        phi_l_arr = (c_double * 1)(0.0)
        phi_r_arr = (c_double * 1)(0.0)
        res_array[0] = phi_l_arr
        res_array[1] = phi_r_arr
        
        # 准备工作空间（如果需要）
        iw = (c_longlong * self.inv_sz_iw)()
        w = (c_double * self.inv_sz_w)()
        
        # 调用函数
        ret = self.lib.ankle_inv(arg_array, res_array, iw, w, 0)
        
        if ret != 0:
            raise RuntimeError(f"ankle_inv调用失败，返回码: {ret}")
        
        return float(phi_l_arr[0]), float(phi_r_arr[0])
    
    def jacobian(self, pitch, roll, d1=None, d2=None, h1=None, h2=None,
                 r1=None, r2=None, u_x=None, u_z=None):
        """
        调用C版本的雅可比矩阵计算函数
        
        @param pitch: 俯仰角
        @param roll: 滚转角
        @param d1, d2, h1, h2, r1, r2, u_x, u_z: 几何参数（可选，默认使用AnkleInfo的值）
        @return: 2x2雅可比矩阵（numpy数组）
        """
        # 如果参数未提供，使用AnkleInfo中的值
        if d1 is None: d1 = self.info.d1
        if d2 is None: d2 = self.info.d2
        if h1 is None: h1 = self.info.h1
        if h2 is None: h2 = self.info.h2
        if r1 is None: r1 = self.info.r1
        if r2 is None: r2 = self.info.r2
        if u_x is None: u_x = self.info.u_x
        if u_z is None: u_z = self.info.u_z
        
        # 准备输入数组
        inputs = [pitch, roll, d1, d2, h1, h2, r1, r2, u_x, u_z]
        arg_array = (POINTER(c_double) * self.jac_sz_arg)()
        for i, val in enumerate(inputs):
            arg_array[i] = (c_double * 1)(val)
        
        # 准备输出数组（2x2矩阵，按行展开）
        res_array = (POINTER(c_double) * self.jac_sz_res)()
        jac_arr = (c_double * 4)(0.0)  # 2x2 = 4个元素
        res_array[0] = jac_arr
        
        # 准备工作空间
        iw = (c_longlong * self.jac_sz_iw)()
        w = (c_double * self.jac_sz_w)()
        
        # 调用函数
        ret = self.lib.ankle_jacobian(arg_array, res_array, iw, w, 0)
        
        if ret != 0:
            raise RuntimeError(f"ankle_jacobian调用失败，返回码: {ret}")
        
        # 将结果转换为2x2矩阵
        # CasADi矩阵按列主序存储: [∂phi_l/∂pitch, ∂phi_r/∂pitch, ∂phi_l/∂roll, ∂phi_r/∂roll]
        # 即: jac_arr = [J[0,0], J[1,0], J[0,1], J[1,1]]
        jac_matrix = np.array([
            [jac_arr[0], jac_arr[2]],  # [∂phi_l/∂pitch, ∂phi_l/∂roll]
            [jac_arr[1], jac_arr[3]]   # [∂phi_r/∂pitch, ∂phi_r/∂roll]
        ])
        
        return jac_matrix
    
    def forward(self, q_m, q_j0, d1=None, d2=None, h1=None, h2=None,
                r1=None, r2=None, u_x=None, u_z=None, max_iter=100, tol=1e-4, alpha=0.9):
        """
        调用C版本的正向运动学函数（使用牛顿迭代法）
        
        @param q_m: 目标关节角度 [phi_l, phi_r]
        @param q_j0: 初始猜测的姿态 [pitch, roll]
        @param d1, d2, h1, h2, r1, r2, u_x, u_z: 几何参数（可选，默认使用AnkleInfo的值）
        @param max_iter: 最大迭代次数（默认100）
        @param tol: 收敛容差（默认1e-4）
        @param alpha: 步长因子（默认0.9）
        @return: (q_j, iter_count, err) 其中q_j是求解得到的姿态[pitch, roll]
        """
        # 如果参数未提供，使用AnkleInfo中的值
        if d1 is None: d1 = self.info.d1
        if d2 is None: d2 = self.info.d2
        if h1 is None: h1 = self.info.h1
        if h2 is None: h2 = self.info.h2
        if r1 is None: r1 = self.info.r1
        if r2 is None: r2 = self.info.r2
        if u_x is None: u_x = self.info.u_x
        if u_z is None: u_z = self.info.u_z
        
        q_m = np.array(q_m)
        q_j = np.array(q_j0, dtype=np.float64)
        
        # 计算初始关节角度
        phi_l0, phi_r0 = self.inv(q_j[0], q_j[1], d1, d2, h1, h2, r1, r2, u_x, u_z)
        q_m0 = np.array([phi_l0, phi_r0])
        err = np.linalg.norm(q_m - q_m0)
        iter_count = 0
        
        while err > tol and iter_count < max_iter:
            # 计算雅可比矩阵
            J = self.jacobian(q_j[0], q_j[1], d1, d2, h1, h2, r1, r2, u_x, u_z)
            
            # 使用最快的2x2矩阵求逆方法
            J_inv = fast_2x2_inverse(J)
            
            # 计算误差
            delta_q_m = q_m - q_m0
            
            # 更新姿态
            q_j = q_j + alpha * (J_inv @ delta_q_m)
            
            # 重新计算关节角度
            phi_l_new, phi_r_new = self.inv(q_j[0], q_j[1], d1, d2, h1, h2, r1, r2, u_x, u_z)
            q_m0 = np.array([phi_l_new, phi_r_new])
            
            # 更新误差
            err = np.linalg.norm(delta_q_m)
            iter_count += 1
        
        return q_j, iter_count, err


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

    # 默认导出带参数的C++函数
    print("导出带参数的C++函数...")
    c_path = ankle.export_cpp("./cpp_version/autogen_code", prefix="ankle")
    
    # 编译为共享库并在Python中调用
    print("\n" + "="*50)
    print("测试Python调用C函数:")
    try:
        lib_path = ankle.compile_shared_library(c_path)
        
        # 创建C包装器
        c_wrapper = AnkleCWrapper(lib_path)
        
        # 测试逆运动学
        pitch_test, roll_test = 0.1, 0.2
        phi_l_c, phi_r_c = c_wrapper.inv(pitch_test, roll_test)
        phi_l_py, phi_r_py = ankle.inv(pitch_test, roll_test, return_float=True)
        
        print(f"\n逆运动学测试 (pitch={pitch_test}, roll={roll_test}):")
        print(f"  Python版本: phi_l={phi_l_py:.6f}, phi_r={phi_r_py:.6f}")
        print(f"  C版本:      phi_l={phi_l_c:.6f}, phi_r={phi_r_c:.6f}")
        print(f"  误差:        phi_l={abs(phi_l_py-phi_l_c):.2e}, phi_r={abs(phi_r_py-phi_r_c):.2e}")
        
        # 测试雅可比矩阵
        jac_c = c_wrapper.jacobian(pitch_test, roll_test)
        jac_py = ankle.jacobian(roll_test, pitch_test, return_numpy=True)
        
        print(f"\n雅可比矩阵测试:")
        print(f"  Python版本:\n{jac_py}")
        print(f"  C版本:\n{jac_c}")
        print(f"  误差:\n{np.abs(jac_py - jac_c)}")
        
    except Exception as e:
        print(f"编译或调用C函数失败: {e}")
        import traceback
        traceback.print_exc()
    
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
    # inv
    idx = 0
    use_last_q_j = False # 是否使用上一次的q_j作为初始值，但是此处实验的采样方法存在跳变，所以使用这个方法会导致部分点无法收敛
    solved_phi = []
    for pitch in range(-80, 80, 5):
        pitch = 0.01 * pitch
        for roll in range(-80, 80, 5):
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
                solved_phi.append([phi_l, phi_r])
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
    # 散点图
    solved_phi = np.array(solved_phi)*180/np.pi
    plt.scatter(solved_phi[:, 0], solved_phi[:, 1], c=solved_phi[:, 0], cmap='viridis', alpha=0.7, s=20)
    plt.xlabel('phi_l (deg)')
    plt.ylabel('phi_r (deg)')
    plt.title('Solved Phi Points')
    plt.grid(True)
    plt.show()
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
    # 正向运动学求解空间验证
    print('='*50)
    print("开始正向运动学求解空间验证...")
    
    workspace_validation_results = []
    workspace_validation_fail = []
    
    # 关节角度范围：-40°到40°
    joint_range_deg = 40
    joint_range_rad = np.radians(joint_range_deg)
    
    # 采样密度（可以调整）
    sample_density = 5  # 每5度采样一次
    
    idx_workspace = 0
    for phi_l_deg in range(-joint_range_deg, joint_range_deg + 1, sample_density):
        phi_l_rad = np.radians(phi_l_deg)
        for phi_r_deg in range(-joint_range_deg, joint_range_deg + 1, sample_density):
            phi_r_rad = np.radians(phi_r_deg)
            
            q_m = np.array([phi_l_rad, phi_r_rad])
            q_j_init = np.array([0.0, 0.0])  # 初始猜测值
            
            try:
                q_j, iter_count, err = ankle.forward(q_m, q_j_init)
                
                if err < 1e-3:
                    # 验证求解结果：用求解得到的pitch, roll重新计算逆运动学
                    pitch_solved, roll_solved = q_j[0], q_j[1]
                    phi_l_verify, phi_r_verify = ankle.inv(pitch_solved, roll_solved, return_float=True)
                    
                    # 计算验证误差
                    verify_err = np.linalg.norm([phi_l_rad - phi_l_verify, phi_r_rad - phi_r_verify])
                    
                    if verify_err < 1e-3:
                        workspace_validation_results.append([phi_l_deg, phi_r_deg, pitch_solved, roll_solved, iter_count, err, verify_err])
                    else:
                        workspace_validation_fail.append([phi_l_deg, phi_r_deg, iter_count, err, verify_err])
                else:
                    workspace_validation_fail.append([phi_l_deg, phi_r_deg, iter_count, err, np.nan])
                    
            except Exception as e:
                workspace_validation_fail.append([phi_l_deg, phi_r_deg, -1, np.nan, np.nan])
                print(f"求解失败 phi_l={phi_l_deg}°, phi_r={phi_r_deg}°: {e}")
            
            idx_workspace += 1
    
    # 转换为numpy数组
    workspace_validation_results = np.array(workspace_validation_results)
    workspace_validation_fail = np.array(workspace_validation_fail)
    
    print(f"正向运动学求解空间验证完成: 成功 {len(workspace_validation_results)}, 失败 {len(workspace_validation_fail)}")
    
    # 绘制求解空间散点图
    if len(workspace_validation_results) > 0:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        phi_l_success = workspace_validation_results[:, 0]
        phi_r_success = workspace_validation_results[:, 1]
        pitch_solved = workspace_validation_results[:, 2]
        roll_solved = workspace_validation_results[:, 3]
        iter_count = workspace_validation_results[:, 4]
        error_vals = workspace_validation_results[:, 5]
        verify_error = workspace_validation_results[:, 6]
        
        # 1. 关节空间散点图（成功案例）
        sc1 = ax1.scatter(phi_l_success, phi_r_success, c=error_vals, cmap='viridis', alpha=0.7, s=20)
        ax1.set_xlabel('phi_l (deg)')
        ax1.set_ylabel('phi_r (deg)')
        ax1.set_title('Forward Kinematics Workspace (Successful Cases)')
        ax1.grid(True)
        ax1.set_xlim(-joint_range_deg, joint_range_deg)
        ax1.set_ylim(-joint_range_deg, joint_range_deg)
        plt.colorbar(sc1, ax=ax1, label='Solver Error')
        
        # 2. 求解得到的姿态空间散点图
        sc2 = ax2.scatter(np.degrees(pitch_solved), np.degrees(roll_solved), c=error_vals, cmap='viridis', alpha=0.7, s=20)
        ax2.set_xlabel('Pitch (deg)')
        ax2.set_ylabel('Roll (deg)')
        ax2.set_title('Solved Pose Space')
        ax2.grid(True)
        plt.colorbar(sc2, ax=ax2, label='Solver Error')
        
        # 3. 迭代次数分布
        sc3 = ax3.scatter(phi_l_success, phi_r_success, c=iter_count, cmap='plasma', alpha=0.7, s=20)
        ax3.set_xlabel('phi_l (deg)')
        ax3.set_ylabel('phi_r (deg)')
        ax3.set_title('Iteration Count Distribution')
        ax3.grid(True)
        ax3.set_xlim(-joint_range_deg, joint_range_deg)
        ax3.set_ylim(-joint_range_deg, joint_range_deg)
        plt.colorbar(sc3, ax=ax3, label='Iterations')
        
        # 4. 验证误差分布
        sc4 = ax4.scatter(phi_l_success, phi_r_success, c=verify_error, cmap='Reds', alpha=0.7, s=20)
        ax4.set_xlabel('phi_l (deg)')
        ax4.set_ylabel('phi_r (deg)')
        ax4.set_title('Verification Error Distribution')
        ax4.grid(True)
        ax4.set_xlim(-joint_range_deg, joint_range_deg)
        ax4.set_ylim(-joint_range_deg, joint_range_deg)
        plt.colorbar(sc4, ax=ax4, label='Verification Error')
        
        plt.tight_layout()
        plt.show()
        
        # 统计信息
        print(f"求解空间统计:")
        print(f"  关节角度范围: phi_l, phi_r ∈ [-{joint_range_deg}°, {joint_range_deg}°]")
        print(f"  采样密度: {sample_density}°")
        print(f"  总采样点数: {idx_workspace}")
        print(f"  成功求解点数: {len(workspace_validation_results)}")
        print(f"  成功率: {len(workspace_validation_results)/idx_workspace*100:.1f}%")
        print(f"  平均迭代次数: {np.mean(iter_count):.1f}")
        print(f"  最大迭代次数: {np.max(iter_count)}")
        print(f"  平均求解误差: {np.mean(error_vals):.2e}")
        print(f"  最大求解误差: {np.max(error_vals):.2e}")
        print(f"  平均验证误差: {np.mean(verify_error):.2e}")
        print(f"  最大验证误差: {np.max(verify_error):.2e}")
        
        # 求解得到的姿态空间范围
        print(f"  求解得到的姿态空间范围:")
        print(f"    Pitch: [{np.degrees(np.min(pitch_solved)):.1f}°, {np.degrees(np.max(pitch_solved)):.1f}°]")
        print(f"    Roll: [{np.degrees(np.min(roll_solved)):.1f}°, {np.degrees(np.max(roll_solved)):.1f}°]")
    
    # 绘制失败案例
    if len(workspace_validation_fail) > 0:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        phi_l_fail = workspace_validation_fail[:, 0]
        phi_r_fail = workspace_validation_fail[:, 1]
        error_fail = workspace_validation_fail[:, 3]
        
        # 过滤掉NaN值
        valid_mask = ~np.isnan(error_fail)
        phi_l_fail_valid = phi_l_fail[valid_mask]
        phi_r_fail_valid = phi_r_fail[valid_mask]
        error_fail_valid = error_fail[valid_mask]
        
        if len(phi_l_fail_valid) > 0:
            sc = ax.scatter(phi_l_fail_valid, phi_r_fail_valid, c=error_fail_valid, cmap='Reds', alpha=0.8, s=30)
            ax.set_xlabel('phi_l (deg)')
            ax.set_ylabel('phi_r (deg)')
            ax.set_title('Failed Cases in Joint Space')
            ax.grid(True)
            ax.set_xlim(-joint_range_deg, joint_range_deg)
            ax.set_ylim(-joint_range_deg, joint_range_deg)
            plt.colorbar(sc, label='Solver Error')
        else:
            ax.text(0.5, 0.5, 'No valid failed cases to plot', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Failed Cases in Joint Space')
        
        plt.tight_layout()
        plt.show()
        
        print(f"  失败求解点数: {len(workspace_validation_fail)}")
        print(f"  失败率: {len(workspace_validation_fail)/idx_workspace*100:.1f}%")


if __name__ == "__main__":
    __main__()
