#ifndef ANKLE_SOLVER_H
#define ANKLE_SOLVER_H

#include <Eigen/Dense>
#include <vector>
#include <tuple>

// CasADi 数据类型定义
#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int long long int
#endif

// 前向声明 CasADi 生成的函数
extern "C" {
    int ankle_inv(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
    int ankle_jacobian(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
}

/**
 * @brief 踝关节运动学求解器 C++ 包装类
 * 
 * 该类封装了由 CasADi 生成的踝关节逆运动学和雅可比矩阵计算函数，
 * 提供了更易用的 C++ 接口。
 */
class AnkleSolver {
public:
    /**
     * @brief 构造函数
     */
    AnkleSolver();
    
    /**
     * @brief 析构函数
     */
    ~AnkleSolver();
    
    /**
     * @brief 踝关节逆运动学求解
     * 
     * 根据给定的俯仰角和横滚角，计算左右电机的角度
     * 
     * @param pitch 俯仰角 (弧度)
     * @param roll 横滚角 (弧度)
     * @return Eigen::Vector2d [phi_l, phi_r] 左右电机角度 (弧度)
     */
    Eigen::Vector2d inverseKinematics(double pitch, double roll);
    
    /**
     * @brief 踝关节逆运动学求解 (Eigen输入)
     * 
     * @param pose Eigen::Vector2d [pitch, roll] 俯仰角和横滚角 (弧度)
     * @return Eigen::Vector2d [phi_l, phi_r] 左右电机角度 (弧度)
     */
    Eigen::Vector2d inverseKinematics(const Eigen::Vector2d& pose);
    
    /**
     * @brief 计算踝关节雅可比矩阵
     * 
     * 计算电机角度相对于俯仰角和横滚角的雅可比矩阵
     * 
     * @param pitch 俯仰角 (弧度)
     * @param roll 横滚角 (弧度)
     * @return Eigen::Matrix2d 2x2 雅可比矩阵
     */
    Eigen::Matrix2d jacobian(double pitch, double roll);
    
    /**
     * @brief 计算踝关节雅可比矩阵 (Eigen输入)
     * 
     * @param pose Eigen::Vector2d [pitch, roll] 俯仰角和横滚角 (弧度)
     * @return Eigen::Matrix2d 2x2 雅可比矩阵
     */
    Eigen::Matrix2d jacobian(const Eigen::Vector2d& pose);
    
    /**
     * @brief 批量逆运动学求解
     * 
     * @param poses Eigen::MatrixXd (2×N) 每列为一个 [pitch, roll] 姿态
     * @return Eigen::MatrixXd (2×N) 每列为对应的 [phi_l, phi_r] 电机角度
     */
    Eigen::MatrixXd batchInverseKinematics(const Eigen::MatrixXd& poses);
    
    /**
     * @brief 批量雅可比矩阵计算
     * 
     * @param poses Eigen::MatrixXd (2×N) 每列为一个 [pitch, roll] 姿态
     * @return std::vector<Eigen::Matrix2d> 对应的雅可比矩阵数组
     */
    std::vector<Eigen::Matrix2d> batchJacobian(const Eigen::MatrixXd& poses);
    
    /**
     * @brief 踝关节正向运动学求解
     * 
     * 根据给定的左右电机角度，通过雅可比迭代方法求解俯仰角和横滚角
     * 
     * @param phi_l 左电机角度 (弧度)
     * @param phi_r 右电机角度 (弧度)
     * @param initial_guess 初始猜测值 [pitch, roll] (弧度)
     * @param max_iterations 最大迭代次数 (默认100)
     * @param tolerance 收敛容差 (默认1e-6)
     * @param step_size 步长因子 (默认0.9)
     * @return std::tuple<Eigen::Vector2d, int, double> [pose, iterations, error]
     */
    std::tuple<Eigen::Vector2d, int, double> forwardKinematics(
        double phi_l, double phi_r, 
        const Eigen::Vector2d& initial_guess = Eigen::Vector2d::Zero(),
        int max_iterations = 100,
        double tolerance = 1e-6,
        double step_size = 0.9);
    
    /**
     * @brief 踝关节正向运动学求解 (Eigen输入)
     * 
     * @param motors Eigen::Vector2d [phi_l, phi_r] 左右电机角度 (弧度)
     * @param initial_guess 初始猜测值 [pitch, roll] (弧度)
     * @param max_iterations 最大迭代次数 (默认100)
     * @param tolerance 收敛容差 (默认1e-6)
     * @param step_size 步长因子 (默认0.9)
     * @return std::tuple<Eigen::Vector2d, int, double> [pose, iterations, error]
     */
    std::tuple<Eigen::Vector2d, int, double> forwardKinematics(
        const Eigen::Vector2d& motors,
        const Eigen::Vector2d& initial_guess = Eigen::Vector2d::Zero(),
        int max_iterations = 100,
        double tolerance = 1e-6,
        double step_size = 0.9);
    
    /**
     * @brief 批量正向运动学求解
     * 
     * @param motors_batch Eigen::MatrixXd (2×N) 每列为一个 [phi_l, phi_r] 电机角度
     * @param initial_guesses Eigen::MatrixXd (2×N) 每列为对应的初始猜测值，可选
     * @param max_iterations 最大迭代次数 (默认100)
     * @param tolerance 收敛容差 (默认1e-6)
     * @param step_size 步长因子 (默认0.9)
     * @return std::tuple<Eigen::MatrixXd, Eigen::VectorXi, Eigen::VectorXd> [poses, iterations, errors]
     */
    std::tuple<Eigen::MatrixXd, Eigen::VectorXi, Eigen::VectorXd> batchForwardKinematics(
        const Eigen::MatrixXd& motors_batch,
        const Eigen::MatrixXd& initial_guesses = Eigen::MatrixXd(),
        int max_iterations = 100,
        double tolerance = 1e-6,
        double step_size = 0.9);

    /**
     * @brief 关节速度映射到电机速度
     * 
     * @param joint_pos Eigen::Vector2d [pitch, roll] 关节位置 (弧度)
     * @param joint_velocity Eigen::Vector2d [pitch, roll] 关节速度 (弧度/秒)
     * @return Eigen::Vector2d [phi_l, phi_r] 电机速度 (弧度/秒)
     */
    Eigen::Vector2d velocityJoint2motor(const Eigen::Vector2d& joint_pos, const Eigen::Vector2d& joint_velocity);

    /**
     * @brief 电机速度映射到关节速度
     * 
     * @param joint_pos Eigen::Vector2d [pitch, roll] 关节位置 (弧度)
     * @param motor_velocity Eigen::Vector2d [phi_l, phi_r] 电机速度 (弧度/秒)
     * @return Eigen::Vector2d [pitch, roll] 关节速度 (弧度/秒)
     */
    Eigen::Vector2d velocityJMotor2joint(const Eigen::Vector2d& joint_pos, const Eigen::Vector2d& motor_velocity);
    
    /**
     * @brief 关节扭矩映射到电机扭矩
     * 
     * @param joint_pos Eigen::Vector2d [pitch, roll] 关节位置 (弧度)
     * @param joint_torque Eigen::Vector2d [pitch, roll] 关节扭矩 (牛顿米)
     * @return Eigen::Vector2d [phi_l, phi_r] 电机扭矩 (牛顿米)
     */
    Eigen::Vector2d torqueJoint2motor(const Eigen::Vector2d& joint_pos, const Eigen::Vector2d& joint_torque);

    /**
     * @brief 电机扭矩映射到关节扭矩
     * 
     * @param joint_pos Eigen::Vector2d [pitch, roll] 关节位置 (弧度)
     * @param joint_torque Eigen::Vector2d [pitch, roll] 关节扭矩 (牛顿米)
     * @return Eigen::Vector2d [pitch, roll] 关节扭矩 (牛顿米)
     */
    Eigen::Vector2d torqueMotor2joint(const Eigen::Vector2d& joint_pos, const Eigen::Vector2d& joint_torque);

private:
    // 工作内存指针，用于 CasADi 函数调用
    casadi_int* iw_;
    casadi_real* w_;
    
    /**
     * @brief 初始化工作内存
     */
    void initializeWorkspace();
    
    /**
     * @brief 清理工作内存
     */
    void cleanupWorkspace();
    
    /**
     * @brief 快速计算2x2矩阵的逆
     * 
     * @param matrix 2x2矩阵
     * @return Eigen::Matrix2d 逆矩阵
     * @throws std::runtime_error 如果矩阵奇异
     */
    Eigen::Matrix2d fast2x2Inverse(const Eigen::Matrix2d& matrix);
};

#endif // ANKLE_SOLVER_H
