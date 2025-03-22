import casadi as cs
import numpy as np
import matplotlib.pyplot as plt

class RRRKinematics:
    def __init__(self, l1, l2, l3):
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        
        # 定义符号变量
        theta1 = cs.MX.sym('theta1')  # 第一个关节角度
        theta2 = cs.MX.sym('theta2')  # 第二个关节角度
        theta3 = cs.MX.sym('theta3')  # 第三个关节角度
        l1 = cs.MX.sym('l1')          # 第一连杆长度
        l2 = cs.MX.sym('l2')          # 第二连杆长度
        l3 = cs.MX.sym('l3')          # 第三连杆长度

        # 正向运动学表达式
        x1 = l1*cs.cos(theta1)
        y1 = l1*cs.sin(theta1)
        x2 = x1 + l2*cs.cos(theta1 + theta2)
        y2 = y1 + l2*cs.sin(theta1 + theta2)
        x = x2 + l3*cs.cos(theta1 + theta2 + theta3)
        y = y2 + l3*cs.sin(theta1 + theta2 + theta3)
        pos = cs.vertcat(x, y)  # 末端位置向量

        # 计算雅可比矩阵 (自动微分)
        J = cs.jacobian(pos, cs.vertcat(theta1, theta2, theta3))
        H = cs.jacobian(J, cs.vertcat(theta1, theta2, theta3))

        # 创建可调用函数
        self.kinematics = cs.Function('fk', 
                                [theta1, theta2, theta3, l1, l2, l3], 
                                [pos, J, H],
                                ['theta1', 'theta2', 'theta3', 'l1', 'l2', 'l3'],
                                ['position', 'jacobian', 'hessian'])

    def FK(self, q):
        return self.kinematics(q[0], q[1], q[2], self.l1, self.l2, self.l3)[0]
    
    def jacobian(self, q):
        return self.kinematics(q[0], q[1], q[2], self.l1, self.l2, self.l3)[1]
    
    def hessian(self, q):
        return self.kinematics(q[0], q[1], q[2], self.l1, self.l2, self.l3)[2]

    def visualize(self, q, c='r', alpha=0.5):
        # link1
        x1 = self.l1*cs.cos(q[0])
        y1 = self.l1*cs.sin(q[0])
        plt.plot([0, x1], [0, y1], c+'-', alpha=alpha)

        # link2
        x2 = x1 + self.l2*cs.cos(q[0] + q[1])
        y2 = y1 + self.l2*cs.sin(q[0] + q[1])
        plt.plot([x1, x2], [y1, y2], c+'-', alpha=alpha)

        # link3
        x3 = x2 + self.l3*cs.cos(q[0] + q[1] + q[2])
        y3 = y2 + self.l3*cs.sin(q[0] + q[1] + q[2])
        plt.plot([x2, x3], [y2, y3], c+'-', alpha=alpha)

        # 关节点
        plt.plot(x1, y1, c+'o', markersize=10, alpha=alpha)
        plt.plot(x2, y2, c+'o', markersize=10, alpha=alpha)
        plt.plot(x3, y3, c+'o', markersize=10, alpha=alpha)

class RRRKinematicsBias:
    def __init__(self, l1, l2, l3, bias):
        self.rrr = RRRKinematics(l1, l2, l3)
        self.bias = bias

    def FK(self, q):
        q_hat = q + self.bias
        return self.rrr.FK(q_hat)

    def jacobian(self, q):
        q_hat = q + self.bias
        return self.rrr.jacobian(q_hat)
    
    def visualize(self, q, c='r', alpha=0.5):
        q_hat = q + self.bias
        print("q_hat: ", q_hat)
        self.rrr.visualize(q_hat, c, alpha)

if __name__ == "__main__":
    l1, l2, l3 = 1.5, 1.0, 0.8
    rrr = RRRKinematics(l1, l2, l3)
    q = np.array([cs.pi/3, cs.pi/4, cs.pi/6])
    print("FK: ", rrr.FK(q))
    print("Jacobian: ", rrr.jacobian(q))
    
    # true bias, hat{q} = q + bias
    bias = np.array([0.2, 0.3, 0.1])
    rrr_true = RRRKinematicsBias(l1, l2, l3, bias)
    
    plt.figure(figsize=(8, 8))
    rrr.visualize(q, c='r', alpha=0.5)
    rrr_true.visualize(q, c='g', alpha=0.5)
    plt.grid(True)
    plt.axis('equal')
    # plt.show()

    # sample 2 points

    q_hat = q
    for i in range(1000):
        J = rrr.jacobian(q_hat)
        e = rrr.FK(q_hat) - rrr_true.FK(q)
        if np.linalg.norm(e) < 1e-4:
            print("Converged!")
            break
        # grad_e = -J.T @ e
        lambda_ = 1e-3
        grad_e = -np.linalg.inv(J.T @ J + lambda_ * np.eye(3)) @ J.T @ e
        q_hat = q_hat + 0.1 * grad_e
        print(f"iter {i}: error: {e}, q_hat: {q_hat}")
    print("q bias: ", q_hat - q)
    rrr.visualize(np.array(q_hat), c='b', alpha=0.5)
    plt.show()
