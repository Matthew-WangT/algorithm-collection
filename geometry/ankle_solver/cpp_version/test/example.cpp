#include "include/ankle_solver.h"
#include <iostream>
#include <iomanip>
#include <cmath>

void print_vector(const Eigen::Vector2d& vec, const std::string& name) {
    std::cout << name << ": [" << std::setw(10) << vec(0) << ", " << std::setw(10) << vec(1) << "]" << std::endl;
}

void print_matrix(const Eigen::Matrix2d& matrix, const std::string& name) {
    std::cout << name << ":" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    for (int i = 0; i < matrix.rows(); ++i) {
        std::cout << "[";
        for (int j = 0; j < matrix.cols(); ++j) {
            std::cout << std::setw(10) << matrix(i, j);
            if (j < matrix.cols() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
}

int main() {
    try {
        // 创建踝关节求解器实例
        AnkleSolver solver;
        
        std::cout << "=== 踝关节运动学求解器测试 ===" << std::endl;
        
        // 测试单个点的逆运动学
        std::cout << "\n1. 单点逆运动学测试:" << std::endl;
        double pitch = 0.1;  // 俯仰角 (弧度)
        double roll = 0.05;  // 横滚角 (弧度)
        
        std::cout << "输入: pitch = " << pitch << " rad, roll = " << roll << " rad" << std::endl;
        
        auto ik_result = solver.inverse_kinematics(pitch, roll);
        print_vector(ik_result, "输出 [phi_l, phi_r]");
        
        // 测试 Eigen 输入版本
        Eigen::Vector2d pose(pitch, roll);
        auto ik_result_eigen = solver.inverse_kinematics(pose);
        print_vector(ik_result_eigen, "Eigen输入版本结果");
        
        // 测试雅可比矩阵
        std::cout << "\n2. 雅可比矩阵测试:" << std::endl;
        auto jac = solver.jacobian(pitch, roll);
        print_matrix(jac, "雅可比矩阵 (2x2)");
        
        // 测试 Eigen 输入版本的雅可比
        auto jac_eigen = solver.jacobian(pose);
        print_matrix(jac_eigen, "Eigen输入版本雅可比");
        
        // 测试批量计算
        std::cout << "\n3. 批量逆运动学测试:" << std::endl;
        
        // 创建批量输入矩阵 (2×N)
        Eigen::MatrixXd poses(2, 4);
        poses.col(0) << 0.0, 0.0;
        poses.col(1) << 0.1, 0.05;
        poses.col(2) << 0.2, -0.05;
        poses.col(3) << -0.1, 0.1;
        
        auto batch_results = solver.batch_inverse_kinematics(poses);
        
        std::cout << "批量计算结果:" << std::endl;
        std::cout << std::setw(8) << "pitch" << std::setw(8) << "roll" 
                  << std::setw(12) << "phi_l" << std::setw(12) << "phi_r" << std::endl;
        std::cout << std::string(40, '-') << std::endl;
        
        for (int i = 0; i < poses.cols(); ++i) {
            std::cout << std::setw(8) << poses(0, i) 
                      << std::setw(8) << poses(1, i)
                      << std::setw(12) << batch_results(0, i)
                      << std::setw(12) << batch_results(1, i) << std::endl;
        }
        
        // 测试批量雅可比计算
        std::cout << "\n4. 批量雅可比矩阵测试:" << std::endl;
        auto batch_jacobians = solver.batch_jacobian(poses);
        
        for (int i = 0; i < poses.cols(); ++i) {
            std::cout << "姿态 " << i << ": [" << poses(0, i) << ", " << poses(1, i) << "]" << std::endl;
            print_matrix(batch_jacobians[i], "雅可比矩阵");
            std::cout << std::endl;
        }
        
        // 测试边界情况
        std::cout << "\n5. 边界情况测试:" << std::endl;
        std::vector<std::pair<double, double>> boundary_cases = {
            {0.0, 0.0},      // 零位置
            {0.5, 0.0},      // 最大俯仰角
            {0.0, 0.3},      // 最大横滚角
            {-0.3, -0.2}     // 负角度
        };
        
        for (const auto& case_test : boundary_cases) {
            try {
                auto result = solver.inverse_kinematics(case_test.first, case_test.second);
                std::cout << "pitch=" << std::setw(6) << case_test.first 
                          << ", roll=" << std::setw(6) << case_test.second
                          << " -> phi_l=" << std::setw(8) << result(0)
                          << ", phi_r=" << std::setw(8) << result(1) << std::endl;
            } catch (const std::exception& e) {
                std::cout << "pitch=" << std::setw(6) << case_test.first 
                          << ", roll=" << std::setw(6) << case_test.second
                          << " -> 计算失败: " << e.what() << std::endl;
            }
        }
        
        std::cout << "\n=== 测试完成 ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
