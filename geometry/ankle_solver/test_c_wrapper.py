#!/usr/bin/env python3
"""
示例脚本：演示如何导出C文件并在Python中调用

用法:
    python test_c_wrapper.py
"""

import sys
import os
import numpy as np

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ankle_solver import Ankle, AnkleInfo, AnkleCWrapper

def main():
    # 创建ankle实例
    info = AnkleInfo()
    ankle = Ankle(info)
    
    print("="*60)
    print("步骤1: 导出C文件")
    print("="*60)
    
    # 导出C文件
    c_path = ankle.export_cpp("./cpp_version/autogen_code", prefix="ankle")
    print(f"✓ C文件已导出: {c_path}")
    
    print("\n" + "="*60)
    print("步骤2: 编译为共享库")
    print("="*60)
    
    # 编译为共享库
    try:
        lib_path = ankle.compile_shared_library(c_path)
        print(f"✓ 共享库已编译: {lib_path}")
    except Exception as e:
        print(f"✗ 编译失败: {e}")
        return
    
    print("\n" + "="*60)
    print("步骤3: 在Python中调用C函数")
    print("="*60)
    
    # 创建C包装器
    c_wrapper = AnkleCWrapper(lib_path, info)
    
    # 测试用例
    test_cases = [
        (0.0, 0.0, "零位"),
        (0.1, 0.2, "小角度"),
        (-0.2, 0.1, "负pitch"),
        (0.15, -0.15, "负roll"),
    ]
    
    print("\n逆运动学测试:")
    print("-" * 60)
    max_error = 0.0
    
    for pitch, roll, desc in test_cases:
        # Python版本
        phi_l_py, phi_r_py = ankle.inv(pitch, roll, return_float=True)
        
        # C版本
        phi_l_c, phi_r_c = c_wrapper.inv(pitch, roll)
        
        # 计算误差
        err_l = abs(phi_l_py - phi_l_c)
        err_r = abs(phi_r_py - phi_r_c)
        max_error = max(max_error, err_l, err_r)
        
        print(f"\n{desc} (pitch={pitch:.3f}, roll={roll:.3f}):")
        print(f"  Python: phi_l={phi_l_py:8.6f}, phi_r={phi_r_py:8.6f}")
        print(f"  C:      phi_l={phi_l_c:8.6f}, phi_r={phi_r_c:8.6f}")
        print(f"  误差:   phi_l={err_l:.2e}, phi_r={err_r:.2e}")
    
    print(f"\n最大误差: {max_error:.2e}")
    
    print("\n" + "="*60)
    print("雅可比矩阵测试:")
    print("-" * 60)
    
    pitch_test, roll_test = 0.1, 0.2
    jac_py = ankle.jacobian(roll_test, pitch_test, return_numpy=True)
    jac_c = c_wrapper.jacobian(pitch_test, roll_test)
    
    print(f"\n测试点 (pitch={pitch_test}, roll={roll_test}):")
    print(f"\nPython版本:\n{jac_py}")
    print(f"\nC版本:\n{jac_c}")
    print(f"\n误差:\n{abs(jac_py - jac_c)}")
    
    print("\n" + "="*60)
    print("正向运动学测试:")
    print("-" * 60)
    
    # 测试正向运动学
    test_cases_fk = [
        (np.array([-0.2, -0.2]), np.array([0.0, 0.0]), "小角度"),
        (np.array([0.3, 0.3]), np.array([0.0, 0.0]), "中等角度"),
        (np.array([-0.15, 0.15]), np.array([0.0, 0.0]), "不对称角度"),
    ]
    
    print("\n正向运动学测试:")
    print("-" * 60)
    
    for q_m, q_j0, desc in test_cases_fk:
        # Python版本
        q_j_py, iter_py, err_py = ankle.forward(q_m, q_j0)
        
        # C版本
        q_j_c, iter_c, err_c = c_wrapper.forward(q_m, q_j0)
        
        # 计算误差
        err_q_j = np.linalg.norm(q_j_py - q_j_c)
        
        print(f"\n{desc} (目标: phi_l={q_m[0]:.3f}, phi_r={q_m[1]:.3f}):")
        print(f"  Python: pitch={q_j_py[0]:.6f}, roll={q_j_py[1]:.6f}, iter={iter_py}, err={err_py:.2e}")
        print(f"  C:      pitch={q_j_c[0]:.6f}, roll={q_j_c[1]:.6f}, iter={iter_c}, err={err_c:.2e}")
        print(f"  差异:   q_j误差={err_q_j:.2e}")
    
    print("\n" + "="*60)
    n_iter = 100
    print(f"性能对比测试 ({n_iter}次调用):")
    print("-" * 60)
    
    import time
    
    # Python版本性能测试
    q_m_test = np.array([-0.2, -0.2])
    q_j0_test = np.array([0.0, 0.0])
    
    start_time = time.time()
    for _ in range(n_iter):
        ankle.forward(q_m_test, q_j0_test)
    py_time = time.time() - start_time
    
    # C版本性能测试
    start_time = time.time()
    for _ in range(n_iter):
        c_wrapper.forward(q_m_test, q_j0_test)
    c_time = time.time() - start_time
    
    print("正向运动学python-c性能测试:")
    print(f"Python版本: {py_time*1000:.2f} ms ({py_time/n_iter*1e6:.2f} μs/次)")
    print(f"C版本:      {c_time*1000:.2f} ms ({c_time/n_iter*1e6:.2f} μs/次)")
    print(f"加速比:     {py_time/c_time:.2f}x")

    #  测试雅各比函数性能
    print("\n" + "="*60)
    print("雅各比矩阵python-c性能测试:")
    print("-" * 60)
    start_time = time.time()
    for _ in range(n_iter):
        ankle.jacobian(pitch_test, roll_test)
    py_time = time.time() - start_time
    start_time = time.time()
    for _ in range(n_iter):
        c_wrapper.jacobian(pitch_test, roll_test)
    c_time = time.time() - start_time
    print(f"Python版本: {py_time*1000:.2f} ms ({py_time/n_iter*1e6:.2f} μs/次)")
    print(f"C版本:      {c_time*1000:.2f} ms ({c_time/n_iter*1e6:.2f} μs/次)")
    print(f"加速比:     {py_time/c_time:.2f}x")

    #  测试逆运动学python-c性能
    print("\n" + "="*60)
    print("逆运动学python-c性能测试:")
    print("-" * 60)
    start_time = time.time()
    for _ in range(n_iter):
        ankle.inv(pitch_test, roll_test)
    py_time = time.time() - start_time
    start_time = time.time()
    for _ in range(n_iter):
        c_wrapper.inv(pitch_test, roll_test)
    c_time = time.time() - start_time
    print(f"Python版本: {py_time*1000:.2f} ms ({py_time/n_iter*1e6:.2f} μs/次)")
    print(f"C版本:      {c_time*1000:.2f} ms ({c_time/n_iter*1e6:.2f} μs/次)")
    print(f"加速比:     {py_time/c_time:.2f}x")
    
    print("\n" + "="*60)
    print("测试完成!")
    print("="*60)

if __name__ == "__main__":
    main()

