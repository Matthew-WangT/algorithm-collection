#include "ankle_solver.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <random>

void printVector(const Eigen::Vector2d& vec, const std::string& name) {
    std::cout << name << ": [" << std::setw(10) << std::fixed << std::setprecision(6) 
              << vec(0) << ", " << std::setw(10) << vec(1) << "]" << std::endl;
}

void printError(const Eigen::Vector2d& error, const std::string& name) {
    double norm = error.norm();
    std::cout << name << " 误差: [" << std::setw(12) << std::scientific << std::setprecision(3)
              << error(0) << ", " << std::setw(12) << error(1) << "] (范数: " 
              << std::setw(12) << norm << ")" << std::endl;
}

int main() {
    try {
        AnkleSolver solver;
        
        std::cout << "=== 函数等价性测试 ===" << std::endl;
        
        // 设置随机数生成器
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> joint_pos_dist(-0.3, 0.3);  // 关节位置范围
        std::uniform_real_distribution<> velocity_dist(-1.0, 1.0);   // 速度范围
        std::uniform_real_distribution<> torque_dist(-10.0, 10.0);   // 扭矩范围
        
        // 测试用例数量
        const int num_tests = 10;
        
        std::cout << "\n1. 速度映射等价性测试:" << std::endl;
        std::cout << std::string(80, '-') << std::endl;
        
        double max_velocity_error = 0.0;
        for (int i = 0; i < num_tests; ++i) {
            // 生成随机测试数据
            Eigen::Vector2d joint_pos(joint_pos_dist(gen), joint_pos_dist(gen));
            Eigen::Vector2d joint_velocity(velocity_dist(gen), velocity_dist(gen));
            
            std::cout << "\n测试 " << i + 1 << ":" << std::endl;
            printVector(joint_pos, "关节位置");
            printVector(joint_velocity, "关节速度");
            
            // 使用独立函数计算
            auto motor_velocity_independent = solver.velocityJoint2motor(joint_pos, joint_velocity);
            
            // 使用组合函数计算
            auto [motor_pos, motor_velocity_combined, motor_torque_dummy] = 
                solver.joint2motor(joint_pos, joint_velocity, Eigen::Vector2d::Zero());
            
            // 比较结果
            Eigen::Vector2d velocity_error = motor_velocity_independent - motor_velocity_combined;
            
            printVector(motor_velocity_independent, "独立函数结果");
            printVector(motor_velocity_combined, "组合函数结果");
            printError(velocity_error, "速度");
            
            max_velocity_error = std::max(max_velocity_error, velocity_error.norm());
        }
        
        std::cout << "\n速度映射最大误差: " << std::scientific << max_velocity_error << std::endl;
        
        std::cout << "\n2. 扭矩映射等价性测试:" << std::endl;
        std::cout << std::string(80, '-') << std::endl;
        
        double max_torque_error = 0.0;
        for (int i = 0; i < num_tests; ++i) {
            // 生成随机测试数据
            Eigen::Vector2d joint_pos(joint_pos_dist(gen), joint_pos_dist(gen));
            Eigen::Vector2d joint_torque(torque_dist(gen), torque_dist(gen));
            
            std::cout << "\n测试 " << i + 1 << ":" << std::endl;
            printVector(joint_pos, "关节位置");
            printVector(joint_torque, "关节扭矩");
            
            // 使用独立函数计算
            auto motor_torque_independent = solver.torqueJoint2motor(joint_pos, joint_torque);
            
            // 使用组合函数计算
            auto [motor_pos_dummy, motor_velocity_dummy, motor_torque_combined] = 
                solver.joint2motor(joint_pos, Eigen::Vector2d::Zero(), joint_torque);
            
            // 比较结果
            Eigen::Vector2d torque_error = motor_torque_independent - motor_torque_combined;
            
            printVector(motor_torque_independent, "独立函数结果");
            printVector(motor_torque_combined, "组合函数结果");
            printError(torque_error, "扭矩");
            
            max_torque_error = std::max(max_torque_error, torque_error.norm());
        }
        
        std::cout << "\n扭矩映射最大误差: " << std::scientific << max_torque_error << std::endl;
        
        std::cout << "\n3. 反向速度映射等价性测试:" << std::endl;
        std::cout << std::string(80, '-') << std::endl;
        
        double max_reverse_velocity_error = 0.0;
        for (int i = 0; i < num_tests; ++i) {
            // 生成随机测试数据
            Eigen::Vector2d joint_pos(joint_pos_dist(gen), joint_pos_dist(gen));
            Eigen::Vector2d motor_velocity(velocity_dist(gen), velocity_dist(gen));
            
            std::cout << "\n测试 " << i + 1 << ":" << std::endl;
            printVector(joint_pos, "关节位置");
            printVector(motor_velocity, "电机速度");
            
            // 使用独立函数计算
            auto joint_velocity_independent = solver.velocityMotor2joint(joint_pos, motor_velocity);
            
            // 使用组合函数计算 - 先获得对应的电机位置
            auto motor_pos = solver.inverseKinematics(joint_pos);
            auto [joint_pos_recovered, joint_velocity_combined, joint_torque_dummy] = 
                solver.motor2joint(motor_pos, motor_velocity, Eigen::Vector2d::Zero());
            
            // 比较结果
            Eigen::Vector2d velocity_error = joint_velocity_independent - joint_velocity_combined;
            
            printVector(joint_velocity_independent, "独立函数结果");
            printVector(joint_velocity_combined, "组合函数结果");
            printError(velocity_error, "反向速度");
            
            max_reverse_velocity_error = std::max(max_reverse_velocity_error, velocity_error.norm());
        }
        
        std::cout << "\n4. 反向扭矩映射等价性测试:" << std::endl;
        std::cout << std::string(80, '-') << std::endl;
        
        double max_reverse_torque_error = 0.0;
        for (int i = 0; i < num_tests; ++i) {
            // 生成随机测试数据
            Eigen::Vector2d joint_pos(joint_pos_dist(gen), joint_pos_dist(gen));
            Eigen::Vector2d motor_torque(torque_dist(gen), torque_dist(gen));
            
            std::cout << "\n测试 " << i + 1 << ":" << std::endl;
            printVector(joint_pos, "关节位置");
            printVector(motor_torque, "电机扭矩");
            
            // 使用独立函数计算
            auto joint_torque_independent = solver.torqueMotor2joint(joint_pos, motor_torque);
            
            // 使用组合函数计算 - 先获得对应的电机位置
            auto motor_pos = solver.inverseKinematics(joint_pos);
            auto [joint_pos_recovered, joint_velocity_dummy, joint_torque_combined] = 
                solver.motor2joint(motor_pos, Eigen::Vector2d::Zero(), motor_torque);
            
            // 比较结果
            Eigen::Vector2d torque_error = joint_torque_independent - joint_torque_combined;
            
            printVector(joint_torque_independent, "独立函数结果");
            printVector(joint_torque_combined, "组合函数结果");
            printError(torque_error, "反向扭矩");
            
            max_reverse_torque_error = std::max(max_reverse_torque_error, torque_error.norm());
        }
        
        std::cout << "\n5. 完整映射一致性测试:" << std::endl;
        std::cout << std::string(80, '-') << std::endl;
        
        for (int i = 0; i < 5; ++i) {
            // 生成随机测试数据
            Eigen::Vector2d joint_pos(joint_pos_dist(gen), joint_pos_dist(gen));
            Eigen::Vector2d joint_velocity(velocity_dist(gen), velocity_dist(gen));
            Eigen::Vector2d joint_torque(torque_dist(gen), torque_dist(gen));
            
            std::cout << "\n测试 " << i + 1 << ":" << std::endl;
            printVector(joint_pos, "关节位置");
            printVector(joint_velocity, "关节速度");
            printVector(joint_torque, "关节扭矩");
            
            // 使用组合函数进行正向映射
            auto [motor_pos, motor_velocity, motor_torque] = 
                solver.joint2motor(joint_pos, joint_velocity, joint_torque);
            
            std::cout << "\n正向映射结果:" << std::endl;
            printVector(motor_pos, "电机位置");
            printVector(motor_velocity, "电机速度");
            printVector(motor_torque, "电机扭矩");
            
            // 验证位置映射的一致性
            auto motor_pos_ik = solver.inverseKinematics(joint_pos);
            Eigen::Vector2d pos_error = motor_pos - motor_pos_ik;
            printError(pos_error, "位置映射");
            
            // 验证速度映射的一致性
            auto motor_velocity_independent = solver.velocityJoint2motor(joint_pos, joint_velocity);
            Eigen::Vector2d vel_error = motor_velocity - motor_velocity_independent;
            printError(vel_error, "速度映射");
            
            // 验证扭矩映射的一致性
            auto motor_torque_independent = solver.torqueJoint2motor(joint_pos, joint_torque);
            Eigen::Vector2d torque_error = motor_torque - motor_torque_independent;
            printError(torque_error, "扭矩映射");
        }
        
        // 总结结果
        std::cout << "\n=== 测试总结 ===" << std::endl;
        std::cout << "速度映射最大误差: " << std::scientific << max_velocity_error << std::endl;
        std::cout << "扭矩映射最大误差: " << std::scientific << max_torque_error << std::endl;
        std::cout << "反向速度映射最大误差: " << std::scientific << max_reverse_velocity_error << std::endl;
        std::cout << "反向扭矩映射最大误差: " << std::scientific << max_reverse_torque_error << std::endl;
        
        const double tolerance = 1e-6;  // 稍微放宽容差，考虑到正向运动学的迭代误差
        bool velocity_passed = max_velocity_error < tolerance;
        bool torque_passed = max_torque_error < tolerance;
        bool reverse_velocity_passed = max_reverse_velocity_error < tolerance;
        bool reverse_torque_passed = max_reverse_torque_error < tolerance;
        
        std::cout << "\n等价性测试结果:" << std::endl;
        std::cout << "正向速度映射等价性: " << (velocity_passed ? "通过" : "失败") << std::endl;
        std::cout << "正向扭矩映射等价性: " << (torque_passed ? "通过" : "失败") << std::endl;
        std::cout << "反向速度映射等价性: " << (reverse_velocity_passed ? "通过" : "失败") << std::endl;
        std::cout << "反向扭矩映射等价性: " << (reverse_torque_passed ? "通过" : "失败") << std::endl;
        
        bool all_passed = velocity_passed && torque_passed && reverse_velocity_passed && reverse_torque_passed;
        
        if (all_passed) {
            std::cout << "\n✅ 所有测试通过！独立函数与组合函数在数值精度范围内等价。" << std::endl;
        } else {
            std::cout << "\n❌ 部分测试失败，存在数值差异。" << std::endl;
            if (!reverse_velocity_passed || !reverse_torque_passed) {
                std::cout << "注意：反向映射测试可能由于正向运动学的迭代误差而产生较大误差。" << std::endl;
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
