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
        
        // 测试正向运动学
        std::cout << "\n6. 正向运动学测试:" << std::endl;
        
        // 使用已知的逆运动学结果进行测试
        double test_pitch = 0.1;
        double test_roll = 0.05;
        
        // 先计算逆运动学得到电机角度
        auto motors = solver.inverse_kinematics(test_pitch, test_roll);
        std::cout << "原始姿态: pitch=" << test_pitch << ", roll=" << test_roll << std::endl;
        std::cout << "对应电机角度: phi_l=" << motors(0) << ", phi_r=" << motors(1) << std::endl;
        
        // 使用正向运动学恢复姿态
        Eigen::Vector2d initial_guess(0.0, 0.0);  // 零初始猜测
        auto [recovered_pose, iterations, error] = solver.forward_kinematics(
            motors(0), motors(1), initial_guess);
        
        std::cout << "正向运动学恢复的姿态: pitch=" << recovered_pose(0) 
                  << ", roll=" << recovered_pose(1) << std::endl;
        std::cout << "迭代次数: " << iterations << ", 最终误差: " << error << std::endl;
        
        // 计算恢复精度
        double pitch_error = std::abs(test_pitch - recovered_pose(0));
        double roll_error = std::abs(test_roll - recovered_pose(1));
        std::cout << "恢复精度: pitch误差=" << pitch_error 
                  << ", roll误差=" << roll_error << std::endl;
        
        // 测试 Eigen 输入版本
        std::cout << "\n7. 正向运动学 (Eigen输入) 测试:" << std::endl;
        auto [recovered_pose2, iterations2, error2] = solver.forward_kinematics(
            motors, initial_guess);
        
        print_vector(recovered_pose2, "Eigen输入版本恢复姿态");
        std::cout << "迭代次数: " << iterations2 << ", 最终误差: " << error2 << std::endl;
        
        // 测试批量正向运动学
        std::cout << "\n8. 批量正向运动学测试:" << std::endl;
        
        // 使用之前的批量逆运动学结果
        auto [batch_poses, batch_iterations, batch_errors] = solver.batch_forward_kinematics(
            batch_results);
        
        std::cout << "批量正向运动学结果:" << std::endl;
        std::cout << std::setw(8) << "phi_l" << std::setw(8) << "phi_r" 
                  << std::setw(10) << "pitch" << std::setw(10) << "roll"
                  << std::setw(6) << "iter" << std::setw(12) << "error" << std::endl;
        std::cout << std::string(54, '-') << std::endl;
        
        for (int i = 0; i < batch_results.cols(); ++i) {
            std::cout << std::setw(8) << batch_results(0, i)
                      << std::setw(8) << batch_results(1, i)
                      << std::setw(10) << batch_poses(0, i)
                      << std::setw(10) << batch_poses(1, i)
                      << std::setw(6) << batch_iterations(i)
                      << std::setw(12) << batch_errors(i) << std::endl;
        }
        
        // 验证正向-逆向运动学的一致性
        std::cout << "\n9. 正向-逆向运动学一致性验证:" << std::endl;
        std::vector<std::pair<double, double>> test_cases = {
            {0.0, 0.0},
            {0.1, 0.05},
            {-0.05, 0.1},
            {-0.90, 0.90},
            {-0.91, 0.34},
            {-0.41, 0.93},
            {0.15, -0.08}
        };
        
        std::cout << std::setw(8) << "pitch" << std::setw(8) << "roll"
                  << std::setw(10) << "phi_l" << std::setw(10) << "phi_r"
                  << std::setw(12) << "rec_pitch" << std::setw(12) << "rec_roll"
                  << std::setw(8) << "p_err" << std::setw(8) << "r_err" << std::endl;
        std::cout << std::string(80, '-') << std::endl;
        
        for (const auto& test_case : test_cases) {
            double orig_pitch = test_case.first;
            double orig_roll = test_case.second;
            
            // 逆运动学: 姿态 -> 电机角度
            auto test_motors = solver.inverse_kinematics(orig_pitch, orig_roll);
            
            // 正向运动学: 电机角度 -> 姿态
            auto [rec_pose, iter, err] = solver.forward_kinematics(
                test_motors, Eigen::Vector2d::Zero());
            
            double p_error = std::abs(orig_pitch - rec_pose(0));
            double r_error = std::abs(orig_roll - rec_pose(1));
            
            std::cout << std::setw(8) << orig_pitch
                      << std::setw(8) << orig_roll
                      << std::setw(10) << test_motors(0)
                      << std::setw(10) << test_motors(1)
                      << std::setw(12) << rec_pose(0)
                      << std::setw(12) << rec_pose(1)
                      << std::setw(8) << p_error
                      << std::setw(8) << r_error << std::endl;
        }
        
        std::cout << "\n=== 测试完成 ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
