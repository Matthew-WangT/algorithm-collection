from arm_kinematics import ArmKinematics

import pinocchio as pin
import numpy as np
import os
import nlopt
import time

def create_objective_function(arm_kinematics, q_list, true_bias):
    def objective_function(delta, grad):
        total_error = 0.0
        if grad.size > 0:
            grad[:] = 0.0
        delta = np.array(delta).flatten()
        # true_bias = np.array(true_bias).flatten()

        for q in q_list:
            q = np.array(q).flatten()
            pred_pos, pred_rot, J = arm_kinematics.FK(q + delta)
            true_pos, true_rot, _ = arm_kinematics.FK(q+true_bias)
            
            error_pos = pred_pos - true_pos
            error_rot = pin.log3(pred_rot @ true_rot.T)
            error = np.concatenate([error_pos, error_rot])
            total_error += np.sum(error**2)

            if grad.size > 0:
                grad += 2 * J.T @ error
        
        return float(total_error)
    return objective_function

def main():
    urdf_path = os.path.join(os.path.dirname(__file__), "robot.urdf")
    arm_kinematics = ArmKinematics(urdf_path)

    n_poses = 20
    n_var = 7
    np.random.seed(0)
    q_list = [np.random.uniform(-np.pi, np.pi, n_var) for _ in range(n_poses)]
    print("q_list: ", q_list)
    true_bias = np.array([0.02, -0.03, 0.15, 0.02, 0.03, -0.05, 0.02])
    
    # opt = nlopt.opt(nlopt.LD_SLSQP, n_var)
    # opt = nlopt.opt(nlopt.LD_LBFGS, n_var)
    opt = nlopt.opt(nlopt.LD_MMA, n_var)
    # opt = nlopt.opt(nlopt.GN_ISRES, n_var)
    opt.set_lower_bounds([-10*np.pi/180]*n_var)
    opt.set_upper_bounds([+10*np.pi/180]*n_var)
    
    objective = create_objective_function(arm_kinematics, q_list, true_bias)
    opt.set_min_objective(objective)
    
    opt.set_ftol_rel(1e-8)
    opt.set_maxeval(1000)
    
    initial_delta = np.zeros(n_var)
    
    try:
        time_start = time.time()
        result = opt.optimize(initial_delta)
        time_end = time.time()
        print("找到的偏差：", result)
        print("最终误差：", opt.last_optimum_value())
        print(f"优化时间：{1e3*(time_end - time_start):.2f} 毫秒")
    except Exception as e:
        print("优化错误：", e)

if __name__ == "__main__":
    main()
