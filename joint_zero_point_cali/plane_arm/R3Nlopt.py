import casadi as cs
import numpy as np
import matplotlib.pyplot as plt
import nlopt

class RRRKinematics:
    def __init__(self, l1, l2, l3):
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        
        theta1 = cs.MX.sym('theta1')
        theta2 = cs.MX.sym('theta2')
        theta3 = cs.MX.sym('theta3')
        l1 = cs.MX.sym('l1')
        l2 = cs.MX.sym('l2')
        l3 = cs.MX.sym('l3')

        x1 = l1*cs.cos(theta1)
        y1 = l1*cs.sin(theta1)
        x2 = x1 + l2*cs.cos(theta1 + theta2)
        y2 = y1 + l2*cs.sin(theta1 + theta2)
        x = x2 + l3*cs.cos(theta1 + theta2 + theta3)
        y = y2 + l3*cs.sin(theta1 + theta2 + theta3)
        pos = cs.vertcat(x, y)

        J = cs.jacobian(pos, cs.vertcat(theta1, theta2, theta3))
        H = cs.jacobian(J, cs.vertcat(theta1, theta2, theta3))

        self.kinematics = cs.Function('fk', 
                                [theta1, theta2, theta3, l1, l2, l3], 
                                [pos, J, H],
                                ['theta1', 'theta2', 'theta3', 'l1', 'l2', 'l3'],
                                ['position', 'jacobian', 'hessian'])

    def FK(self, q):
        pos = self.kinematics(q[0], q[1], q[2], self.l1, self.l2, self.l3)[0]
        return np.array(pos.full()).flatten()  # 转换为numpy数组
    
    def jacobian(self, q):
        J = self.kinematics(q[0], q[1], q[2], self.l1, self.l2, self.l3)[1]
        return np.array(J.full())  # 转换为numpy数组
    
    def hessian(self, q):
        H = self.kinematics(q[0], q[1], q[2], self.l1, self.l2, self.l3)[2]
        return np.array(H.full())
    
    def visualize(self, q, c='r', alpha=0.5):
        q = np.array(q).flatten()
        x1 = self.l1 * np.cos(q[0])
        y1 = self.l1 * np.sin(q[0])
        x2 = x1 + self.l2 * np.cos(q[0] + q[1])
        y2 = y1 + self.l2 * np.sin(q[0] + q[1])
        x3 = x2 + self.l3 * np.cos(q[0] + q[1] + q[2])
        y3 = y2 + self.l3 * np.sin(q[0] + q[1] + q[2])
        
        plt.plot([0, x1], [0, y1], c+'-', alpha=alpha)
        plt.plot([x1, x2], [y1, y2], c+'-', alpha=alpha)
        plt.plot([x2, x3], [y2, y3], c+'-', alpha=alpha)
        plt.plot(x1, y1, c+'o', alpha=alpha)
        plt.plot(x2, y2, c+'o', alpha=alpha)
        plt.plot(x3, y3, c+'o', alpha=alpha)

class RRRKinematicsBias:
    def __init__(self, l1, l2, l3, bias):
        self.rrr = RRRKinematics(l1, l2, l3)
        self.bias = np.array(bias).flatten()
    
    def FK(self, q):
        q_hat = np.array(q).flatten() + self.bias
        return self.rrr.FK(q_hat)
    
    def jacobian(self, q):
        q_hat = np.array(q).flatten() + self.bias
        return self.rrr.jacobian(q_hat)
    
    def visualize(self, q, c='r', alpha=0.5):
        q_hat = np.array(q).flatten() + self.bias
        self.rrr.visualize(q_hat, c, alpha)

def create_objective_function(rrr, rrr_true, q_list):
    def objective_function(delta, grad):
        total_error = 0.0
        if grad.size > 0:
            grad[:] = 0.0
        
        delta = np.array(delta).flatten()
        
        for q in q_list:
            q = np.array(q).flatten()
            pred_pos = rrr.FK(q + delta)
            true_pos = rrr_true.FK(q)
            error = pred_pos - true_pos
            total_error += np.sum(error**2)
            
            if grad.size > 0:
                J = rrr.jacobian(q + delta)
                grad += 2 * J.T @ error
        
        return float(total_error)
    return objective_function

def main():
    l1, l2, l3 = 1.5, 1.0, 0.8
    rrr = RRRKinematics(l1, l2, l3)
    true_bias = np.array([0.02, 0.03, 0.05])
    rrr_true = RRRKinematicsBias(l1, l2, l3, true_bias)
    
    n_poses = 10
    q_list = [np.random.uniform(-np.pi, np.pi, 3) for _ in range(n_poses)]
    
    opt = nlopt.opt(nlopt.LD_SLSQP, 3)
    opt.set_lower_bounds([-np.pi]*3)
    opt.set_upper_bounds([np.pi]*3)
    
    objective = create_objective_function(rrr, rrr_true, q_list)
    opt.set_min_objective(objective)
    
    opt.set_ftol_rel(1e-8)
    opt.set_maxeval(1000)
    
    initial_delta = np.zeros(3)
    
    try:
        result = opt.optimize(initial_delta)
        print("找到的偏差：", result)
        print("最终误差：", opt.last_optimum_value())
        
        plt.figure(figsize=(12, 8))
        for q in q_list:
            rrr.visualize(q, c='r', alpha=0.3)
            rrr_true.visualize(q, c='g', alpha=0.3)
            rrr.visualize(q + result, c='b', alpha=0.3)
        
        plt.grid(True)
        plt.axis('equal')
        plt.title('Red: Original  Green: True  Blue: Optimized')
        plt.show()
        
    except Exception as e:
        print("优化错误：", e)

if __name__ == "__main__":
    main()