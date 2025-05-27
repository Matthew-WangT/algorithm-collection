import numpy as np

class RRRobot:
    def __init__(self, L1, L2, theta1_range, theta2_range):
        """
        初始化RR机械臂
        L1: 第一段连杆长度
        L2: 第二段连杆长度
        theta1_range: 第一关节角度范围
        theta2_range: 第二关节角度范围
        """
        self.L1 = L1
        self.L2 = L2
        self.theta1_range = theta1_range
        self.theta2_range = theta2_range
    
    def forward_kinematics(self, theta1, theta2):
        """
        正运动学：计算末端执行器位置
        theta1: 第一关节角度 (弧度)
        theta2: 第二关节角度 (弧度)
        返回: (x, y) 末端执行器位置
        """
        x = self.L1 * np.cos(theta1) + self.L2 * np.cos(theta1 + theta2)
        y = self.L1 * np.sin(theta1) + self.L2 * np.sin(theta1 + theta2)
        return x, y
    
    def get_link_positions(self, theta1, theta2):
        """
        获取所有连杆的位置点
        返回: 基座、关节1、末端执行器的位置
        """
        # 基座位置
        base = (0, 0)
        
        # 第一关节位置
        joint1 = (self.L1 * np.cos(theta1), self.L1 * np.sin(theta1))
        
        # 末端执行器位置
        end_effector = self.forward_kinematics(theta1, theta2)
        
        return base, joint1, end_effector
    
    def check_collision_with_circle(self, theta1, theta2, obstacle_center, obstacle_radius):
        """
        检查机械臂是否与圆形障碍物碰撞
        """
        base, joint1, end_effector = self.get_link_positions(theta1, theta2)
        
        # 检查连杆1是否与障碍物碰撞
        if self._line_circle_collision(base, joint1, obstacle_center, obstacle_radius):
            return True
        
        # 检查连杆2是否与障碍物碰撞
        if self._line_circle_collision(joint1, end_effector, obstacle_center, obstacle_radius):
            return True
        
        return False
    
    def _line_circle_collision(self, p1, p2, circle_center, radius):
        """
        检查线段与圆的碰撞
        """
        x1, y1 = p1
        x2, y2 = p2
        cx, cy = circle_center
        
        # 线段向量
        dx = x2 - x1
        dy = y2 - y1
        
        # 从线段起点到圆心的向量
        fx = x1 - cx
        fy = y1 - cy
        
        # 二次方程系数
        a = dx * dx + dy * dy
        b = 2 * (fx * dx + fy * dy)
        c = (fx * fx + fy * fy) - radius * radius
        
        discriminant = b * b - 4 * a * c
        
        if discriminant < 0:
            return False
        
        # 计算交点参数
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2 * a)
        t2 = (-b + discriminant) / (2 * a)
        
        # 检查交点是否在线段上
        return (0 <= t1 <= 1) or (0 <= t2 <= 1) or (t1 < 0 and t2 > 1)


    def compute_configuration_space(self, obstacles, theta_resolution=100):
        """
        计算配置空间的可行区域
        robot: RRRobot实例
        obstacles: 障碍物列表，每个障碍物为 (center, radius)
        theta_resolution: 角度分辨率
        """
        # 创建角度网格
        theta1_range_list = np.linspace(self.theta1_range[0], self.theta1_range[1], theta_resolution)
        theta2_range_list = np.linspace(self.theta2_range[0], self.theta2_range[1], theta_resolution)
        
        # 初始化配置空间
        config_space = np.ones((theta_resolution, theta_resolution))
        
        for i, theta1 in enumerate(theta1_range_list):
            for j, theta2 in enumerate(theta2_range_list):
                # 检查当前配置是否与任何障碍物碰撞
                collision = False
                for obstacle_center, obstacle_radius in obstacles:
                    if self.check_collision_with_circle(theta1, theta2, obstacle_center, obstacle_radius):
                        collision = True
                        break
                
                # 如果有碰撞，标记为不可行
                if collision:
                    config_space[i, j] = 0
        
        return config_space