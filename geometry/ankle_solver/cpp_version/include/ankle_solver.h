#ifndef ANKLE_SOLVER_H
#define ANKLE_SOLVER_H

#include <Eigen/Dense>
#include <vector>

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
    Eigen::Vector2d inverse_kinematics(double pitch, double roll);
    
    /**
     * @brief 踝关节逆运动学求解 (Eigen输入)
     * 
     * @param pose Eigen::Vector2d [pitch, roll] 俯仰角和横滚角 (弧度)
     * @return Eigen::Vector2d [phi_l, phi_r] 左右电机角度 (弧度)
     */
    Eigen::Vector2d inverse_kinematics(const Eigen::Vector2d& pose);
    
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
    Eigen::MatrixXd batch_inverse_kinematics(const Eigen::MatrixXd& poses);
    
    /**
     * @brief 批量雅可比矩阵计算
     * 
     * @param poses Eigen::MatrixXd (2×N) 每列为一个 [pitch, roll] 姿态
     * @return std::vector<Eigen::Matrix2d> 对应的雅可比矩阵数组
     */
    std::vector<Eigen::Matrix2d> batch_jacobian(const Eigen::MatrixXd& poses);

private:
    // 工作内存指针，用于 CasADi 函数调用
    casadi_int* iw_;
    casadi_real* w_;
    
    /**
     * @brief 初始化工作内存
     */
    void initialize_workspace();
    
    /**
     * @brief 清理工作内存
     */
    void cleanup_workspace();
};

#endif // ANKLE_SOLVER_H
