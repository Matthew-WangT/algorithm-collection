#include "include/ankle_solver.h"
#include <stdexcept>
#include <cstring>
#include <iostream>

AnkleSolver::AnkleSolver() : iw_(nullptr), w_(nullptr) {
    initializeWorkspace();
}

AnkleSolver::~AnkleSolver() {
    cleanupWorkspace();
}

void AnkleSolver::initializeWorkspace() {
    // CasADi 生成的函数通常不需要额外的工作内存
    // 但我们仍然初始化指针为 nullptr
    iw_ = nullptr;
    w_ = nullptr;
}

void AnkleSolver::cleanupWorkspace() {
    // 清理工作内存（如果有的话）
    if (iw_) {
        delete[] iw_;
        iw_ = nullptr;
    }
    if (w_) {
        delete[] w_;
        w_ = nullptr;
    }
}

Eigen::Vector2d AnkleSolver::inverseKinematics(double pitch, double roll) {
    // 准备输入参数
    casadi_real input_pitch = pitch;
    casadi_real input_roll = roll;
    const casadi_real* arg[2] = {&input_pitch, &input_roll};
    
    // 准备输出结果
    casadi_real phi_l, phi_r;
    casadi_real* res[2] = {&phi_l, &phi_r};
    
    // 调用 CasADi 生成的逆运动学函数
    int result = ankle_inv(arg, res, iw_, w_, 0);
    
    if (result != 0) {
        throw std::runtime_error("踝关节逆运动学计算失败，错误代码: " + std::to_string(result));
    }
    
    return Eigen::Vector2d(phi_l, phi_r);
}

Eigen::Vector2d AnkleSolver::inverseKinematics(const Eigen::Vector2d& pose) {
    return inverseKinematics(pose(0), pose(1));
}

Eigen::Matrix2d AnkleSolver::jacobian(double pitch, double roll) {
    // 准备输入参数
    casadi_real input_pitch = pitch;
    casadi_real input_roll = roll;
    const casadi_real* arg[2] = {&input_pitch, &input_roll};
    
    // 准备输出结果 (2x2 矩阵，按列存储)
    casadi_real jacobian_data[4];
    casadi_real* res[1] = {jacobian_data};
    
    // 调用 CasADi 生成的雅可比矩阵函数
    int result = ankle_jacobian(arg, res, iw_, w_, 0);
    
    if (result != 0) {
        throw std::runtime_error("踝关节雅可比矩阵计算失败，错误代码: " + std::to_string(result));
    }
    
    // 将结果转换为 Eigen 矩阵格式 (CasADi 按列存储)
    Eigen::Matrix2d jac;
    jac(0, 0) = jacobian_data[0];  // ∂phi_l/∂pitch
    jac(1, 0) = jacobian_data[1];  // ∂phi_r/∂pitch
    jac(0, 1) = jacobian_data[2];  // ∂phi_l/∂roll
    jac(1, 1) = jacobian_data[3];  // ∂phi_r/∂roll
    
    return jac;
}

Eigen::Matrix2d AnkleSolver::jacobian(const Eigen::Vector2d& pose) {
    return jacobian(pose(0), pose(1));
}

Eigen::MatrixXd AnkleSolver::batchInverseKinematics(const Eigen::MatrixXd& poses) {
    if (poses.rows() != 2) {
        throw std::invalid_argument("输入矩阵必须是 2×N 格式 (每列为 [pitch, roll])");
    }
    
    int n_poses = poses.cols();
    Eigen::MatrixXd results(2, n_poses);
    
    for (int i = 0; i < n_poses; ++i) {
        try {
            Eigen::Vector2d result = inverseKinematics(poses(0, i), poses(1, i));
            results.col(i) = result;
        } catch (const std::runtime_error& e) {
            throw std::runtime_error("批量计算在索引 " + std::to_string(i) + " 处失败: " + e.what());
        }
    }
    
    return results;
}

std::vector<Eigen::Matrix2d> AnkleSolver::batchJacobian(const Eigen::MatrixXd& poses) {
    if (poses.rows() != 2) {
        throw std::invalid_argument("输入矩阵必须是 2×N 格式 (每列为 [pitch, roll])");
    }
    
    int n_poses = poses.cols();
    std::vector<Eigen::Matrix2d> results;
    results.reserve(n_poses);
    
    for (int i = 0; i < n_poses; ++i) {
        try {
            Eigen::Matrix2d jac = jacobian(poses(0, i), poses(1, i));
            results.push_back(jac);
        } catch (const std::runtime_error& e) {
            throw std::runtime_error("批量雅可比计算在索引 " + std::to_string(i) + " 处失败: " + e.what());
        }
    }
    
    return results;
}

Eigen::Matrix2d AnkleSolver::fast2x2Inverse(const Eigen::Matrix2d& matrix) {
    // 计算行列式
    double det = matrix(0, 0) * matrix(1, 1) - matrix(0, 1) * matrix(1, 0);
    
    if (std::abs(det) < 1e-12) {
        throw std::runtime_error("矩阵奇异，无法求逆");
    }
    
    // 2x2矩阵的逆矩阵公式
    Eigen::Matrix2d inverse;
    inverse(0, 0) =  matrix(1, 1) / det;
    inverse(0, 1) = -matrix(0, 1) / det;
    inverse(1, 0) = -matrix(1, 0) / det;
    inverse(1, 1) =  matrix(0, 0) / det;
    
    return inverse;
}

std::tuple<Eigen::Vector2d, int, double> AnkleSolver::forwardKinematics(
    double phi_l, double phi_r, 
    const Eigen::Vector2d& initial_guess,
    int max_iterations,
    double tolerance,
    double step_size) {
    
    Eigen::Vector2d motors(phi_l, phi_r);
    return forwardKinematics(motors, initial_guess, max_iterations, tolerance, step_size);
}

std::tuple<Eigen::Vector2d, int, double> AnkleSolver::forwardKinematics(
    const Eigen::Vector2d& motors,
    const Eigen::Vector2d& initial_guess,
    int max_iterations,
    double tolerance,
    double step_size) {
    
    // 初始化
    Eigen::Vector2d pose = initial_guess;  // [pitch, roll]
    Eigen::Vector2d target_motors = motors;  // [phi_l, phi_r]
    
    // 计算初始误差
    Eigen::Vector2d current_motors = inverseKinematics(pose);
    Eigen::Vector2d error_vector = target_motors - current_motors;
    double error = error_vector.norm();
    
    int iterations = 0;
    
    while (error > tolerance && iterations < max_iterations) {
        // 计算当前姿态的雅可比矩阵
        Eigen::Matrix2d J = jacobian(pose);
        
        try {
            // 计算雅可比矩阵的逆
            Eigen::Matrix2d J_inv = fast2x2Inverse(J);
            
            // 更新姿态
            Eigen::Vector2d delta_pose = step_size * J_inv * error_vector;
            pose += delta_pose;
            
            // 重新计算误差
            current_motors = inverseKinematics(pose);
            error_vector = target_motors - current_motors;
            error = error_vector.norm();
            
            iterations++;
            
        } catch (const std::runtime_error& e) {
            // 如果雅可比矩阵奇异，尝试使用伪逆
            Eigen::Matrix2d J_pinv = J.completeOrthogonalDecomposition().pseudoInverse();
            
            Eigen::Vector2d delta_pose = step_size * J_pinv * error_vector;
            pose += delta_pose;
            
            current_motors = inverseKinematics(pose);
            error_vector = target_motors - current_motors;
            error = error_vector.norm();
            
            iterations++;
        }
    }
    
    return std::make_tuple(pose, iterations, error);
}

std::tuple<Eigen::MatrixXd, Eigen::VectorXi, Eigen::VectorXd> AnkleSolver::batchForwardKinematics(
    const Eigen::MatrixXd& motors_batch,
    const Eigen::MatrixXd& initial_guesses,
    int max_iterations,
    double tolerance,
    double step_size) {
    
    if (motors_batch.rows() != 2) {
        throw std::invalid_argument("输入矩阵必须是 2×N 格式 (每列为 [phi_l, phi_r])");
    }
    
    int n_poses = motors_batch.cols();
    
    // 检查初始猜测值
    Eigen::MatrixXd guesses;
    if (initial_guesses.size() == 0) {
        // 如果没有提供初始猜测值，使用零向量
        guesses = Eigen::MatrixXd::Zero(2, n_poses);
    } else {
        if (initial_guesses.rows() != 2 || initial_guesses.cols() != n_poses) {
            throw std::invalid_argument("初始猜测值矩阵必须是 2×N 格式，与输入矩阵维度匹配");
        }
        guesses = initial_guesses;
    }
    
    // 准备输出
    Eigen::MatrixXd poses(2, n_poses);
    Eigen::VectorXi iterations(n_poses);
    Eigen::VectorXd errors(n_poses);
    
    // 逐个求解
    for (int i = 0; i < n_poses; ++i) {
        try {
            Eigen::Vector2d motors = motors_batch.col(i);
            Eigen::Vector2d initial_guess = guesses.col(i);
            
            auto [pose, iter, error] = forwardKinematics(
                motors, initial_guess, max_iterations, tolerance, step_size);
            
            poses.col(i) = pose;
            iterations(i) = iter;
            errors(i) = error;
            
        } catch (const std::exception& e) {
            throw std::runtime_error("批量正向运动学计算在索引 " + std::to_string(i) + " 处失败: " + e.what());
        }
    }
    
    return std::make_tuple(poses, iterations, errors);
}

Eigen::Vector2d AnkleSolver::velocityJoint2motor(const Eigen::Vector2d& joint_pos, const Eigen::Vector2d& joint_velocity) {
    Eigen::Matrix2d J = jacobian(joint_pos);
    return J * joint_velocity;
}

Eigen::Vector2d AnkleSolver::velocityJMotor2joint(const Eigen::Vector2d& joint_pos, const Eigen::Vector2d& motor_velocity) {
    Eigen::Matrix2d J = jacobian(joint_pos);
    Eigen::Matrix2d J_inv = fast2x2Inverse(J);
    return J_inv * motor_velocity;
}

Eigen::Vector2d AnkleSolver::torqueJoint2motor(const Eigen::Vector2d& joint_pos, const Eigen::Vector2d& joint_torque) {
    Eigen::Matrix2d J = jacobian(joint_pos);
    Eigen::Matrix2d J_inv = fast2x2Inverse(J);
    return J_inv.transpose() * joint_torque;
}

Eigen::Vector2d AnkleSolver::torqueMotor2joint(const Eigen::Vector2d& joint_pos, const Eigen::Vector2d& motor_torque) {
    Eigen::Matrix2d J = jacobian(joint_pos);
    return J.transpose() * motor_torque;
}