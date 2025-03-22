import casadi as cs
import numpy as np
import matplotlib.pyplot as plt




class RRKinematics:
    def __init__(self, l1, l2):
        self.l1 = l1
        self.l2 = l2
        # 定义符号变量
        theta1 = cs.MX.sym('theta1')  # 第一个关节角度
        theta2 = cs.MX.sym('theta2')  # 第二个关节角度
        l1 = cs.MX.sym('l1')          # 第一连杆长度
        l2 = cs.MX.sym('l2')          # 第二连杆长度

        # 正向运动学表达式
        x = l1*cs.cos(theta1) + l2*cs.cos(theta1 + theta2)
        y = l1*cs.sin(theta1) + l2*cs.sin(theta1 + theta2)
        pos = cs.vertcat(x, y)  # 末端位置向量

        # 计算雅可比矩阵 (自动微分)
        J = cs.jacobian(pos, cs.vertcat(theta1, theta2))
        H = cs.jacobian(J, cs.vertcat(theta1, theta2))

        # 创建可调用函数
        self.kinematics = cs.Function('fk', 
                                [theta1, theta2, l1, l2], 
                                [pos, J, H],
                                ['theta1', 'theta2', 'l1', 'l2'],
                                ['position', 'jacobian', 'hessian'])

    def FK(self, q):
        return self.kinematics(q[0], q[1], self.l1, self.l2)[0]
    
    def jacobian(self, q):
        return self.kinematics(q[0], q[1], self.l1, self.l2)[1]
    
    def hessian(self, q):
        return self.kinematics(q[0], q[1], self.l1, self.l2)[2]

    def visualize(self, q, c='r', alpha=0.5):
        # link1
        x1 = self.l1*cs.cos(q[0])
        y1 = self.l1*cs.sin(q[0])
        # 透明度
        plt.plot([0, x1], [0, y1], c+'-', alpha=alpha)

        # link2
        x2 = x1 + self.l2*cs.cos(q[0] + q[1])
        y2 = y1 + self.l2*cs.sin(q[0] + q[1])
        plt.plot([x1, x2], [y1, y2], c+'-', alpha=alpha)

        # 关节点
        plt.plot(x1, y1, c+'o', markersize=10, alpha=alpha)
        plt.plot(x2, y2, c+'o', markersize=10, alpha=alpha)

        # plt.show()



class RRKinematicsBias:
    def __init__(self, l1, l2, bias):
        self.rr = RRKinematics(l1, l2)
        self.bias = bias

    def FK(self, q):
        q_hat = q + self.bias
        return self.rr.FK(q_hat)

    def jacobian(self, q):
        q_hat = q + self.bias
        return self.rr.jacobian(q_hat)
    
    def visualize(self, q, c='r', alpha=0.5):
        q_hat = q + self.bias
        print("q_hat: ", q_hat)
        self.rr.visualize(q_hat, c, alpha)

if __name__ == "__main__":
    l1, l2 = 1.5, 1
    rr = RRKinematics(l1, l2)
    print("RRKinematics: ", rr.kinematics)
    q = np.array([cs.pi/3, cs.pi/2])
    print("FK: ", rr.FK(q))
    print("Jacobian: ", rr.jacobian(q))
    print("Hessian: ", rr.hessian(q))
    rr.visualize(q, c='r', alpha=0.5)

    # true bias, hat{q} = q + bias
    bias = np.array([0.2, 0.3])
    rr_true = RRKinematicsBias(l1, l2, bias)
    rr_true.visualize(q, c='g', alpha=0.5)
    # plt.show()

    # findout the bias
    bias_guess = np.array([0.0, 0.0])
    q_hat = q + bias_guess

    for i in range(1000):
        # 最速下降法
        # 计算雅可比矩阵
        J = rr.jacobian(q_hat)
        # 计算误差
        e = rr.FK(q_hat) - rr_true.FK(q)
        if np.linalg.norm(e) < 1e-3:
            break
        # print("e: ", e)
        # 计算梯度
        grad_e = -J.T @ e
        # lambda_ = 1e-3
        # grad_e = -np.linalg.inv(J.T @ J + lambda_ * np.eye(2)) @ J.T @ e
        # 更新bias
        bias_guess = bias_guess + 0.1 * grad_e
        print(f"iter {i}: error: {e}, bias_guess: {bias_guess}")
        q_hat = q + bias_guess
    print("q_hat: ", q_hat)
    rr.visualize(np.array(q_hat), c='b', alpha=0.5)
    plt.grid(True)
    plt.axis('equal')
    plt.show()