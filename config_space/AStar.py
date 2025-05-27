import numpy as np
import heapq
from collections import defaultdict

class AStarPlanner:
    def __init__(self, config_space, resolution=100):
        self.config_space = config_space
        self.resolution = resolution
        self.theta_step = 2 * np.pi / resolution
        
    def heuristic(self, a, b):
        """欧几里得距离启发式函数"""
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def get_neighbors(self, node):
        """获取节点的邻居"""
        neighbors = []
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        for dx, dy in directions:
            new_x = node[0] + dx
            new_y = node[1] + dy
            
            # 检查边界
            if 0 <= new_x < self.resolution and 0 <= new_y < self.resolution:
                # 检查是否可行
                if self.config_space[new_x, new_y] == 1:
                    neighbors.append((new_x, new_y))
        
        return neighbors
    
    def config_to_grid(self, theta1, theta2):
        """将配置空间坐标转换为网格坐标"""
        x = int(theta1 / self.theta_step)
        y = int(theta2 / self.theta_step)
        return (min(x, self.resolution-1), min(y, self.resolution-1))
    
    def grid_to_config(self, x, y):
        """将网格坐标转换为配置空间坐标"""
        theta1 = x * self.theta_step
        theta2 = y * self.theta_step
        return (theta1, theta2)
    
    def plan_path(self, start_config, goal_config):
        """使用A*算法规划路径"""
        start = self.config_to_grid(start_config[0], start_config[1])
        goal = self.config_to_grid(goal_config[0], goal_config[1])
        
        # 检查起点和终点是否可行
        if self.config_space[start[0], start[1]] == 0:
            print("起点位置不可行！")
            return None
        if self.config_space[goal[0], goal[1]] == 0:
            print("终点位置不可行！")
            return None
        
        # A*算法
        open_set = [(0, start)]
        came_from = {}
        g_score = defaultdict(lambda: float('inf'))
        g_score[start] = 0
        f_score = defaultdict(lambda: float('inf'))
        f_score[start] = self.heuristic(start, goal)
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if current == goal:
                # 重构路径
                path = []
                while current in came_from:
                    config = self.grid_to_config(current[0], current[1])
                    path.append(config)
                    current = came_from[current]
                config = self.grid_to_config(current[0], current[1])
                path.append(config)
                path.reverse()
                return path
            
            for neighbor in self.get_neighbors(current):
                tentative_g_score = g_score[current] + self.heuristic(current, neighbor)
                
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        print("未找到可行路径！")
        return None