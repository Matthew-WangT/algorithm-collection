import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import matplotlib.patches as patches

from rr_robot import RRRobot


class InteractiveVisualization:
    def __init__(self, robot, obstacles, config_space):
        self.robot = robot
        self.obstacles = obstacles
        self.config_space = config_space
        self.current_theta1 = np.pi/4
        self.current_theta2 = np.pi/4
        
        # 创建图形
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 初始化任务空间图
        self.setup_task_space()
        
        # 初始化配置空间图
        self.setup_config_space()
        
        # 绘制初始机械臂配置
        self.update_robot_display()
        
        # 连接点击事件
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
    def setup_task_space(self):
        """设置任务空间图"""
        self.ax1.set_xlim(-self.robot.L1-self.robot.L2-1, self.robot.L1+self.robot.L2+1)
        self.ax1.set_ylim(-self.robot.L1-self.robot.L2-1, self.robot.L1+self.robot.L2+1)
        self.ax1.set_aspect('equal')
        self.ax1.set_title('Task Space (Click to update configuration)')
        self.ax1.grid(True)
        
        # 绘制机械臂工作空间边界
        theta = np.linspace(0, 2*np.pi, 100)
        # 最大工作半径
        max_reach_x = (self.robot.L1 + self.robot.L2) * np.cos(theta)
        max_reach_y = (self.robot.L1 + self.robot.L2) * np.sin(theta)
        self.ax1.plot(max_reach_x, max_reach_y, 'r--', alpha=0.5, label='Maximum Reachable Area')
        
        # 最小工作半径
        if self.robot.L1 > self.robot.L2:
            min_reach_x = (self.robot.L1 - self.robot.L2) * np.cos(theta)
            min_reach_y = (self.robot.L1 - self.robot.L2) * np.sin(theta)
            self.ax1.plot(min_reach_x, min_reach_y, 'g--', alpha=0.5, label='Minimum Reachable Area')
        
        # 绘制障碍物
        for obstacle_center, obstacle_radius in self.obstacles:
            circle = Circle(obstacle_center, obstacle_radius, color='red', alpha=0.7)
            self.ax1.add_patch(circle)
        
        self.ax1.legend()
        
    def setup_config_space(self):
        """设置配置空间图"""
        im = self.ax2.imshow(self.config_space.T, extent=[0, 2*np.pi, 0, 2*np.pi], 
                            origin='lower', cmap='RdYlGn', aspect='equal')
        self.ax2.set_xlabel('θ₁ (rad)')
        self.ax2.set_ylabel('θ₂ (rad)')
        self.ax2.set_title('Configuration Space (Click to update configuration)')
        
        # 添加刻度标签
        self.ax2.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        self.ax2.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
        self.ax2.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        self.ax2.set_yticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=self.ax2)
        cbar.set_label('Feasibility (1=Feasible, 0=Infeasible)')
        
        # 初始化配置点标记
        self.config_point, = self.ax2.plot(self.current_theta1, self.current_theta2, 
                                          'bo', markersize=10, label='Current Configuration')
        self.ax2.legend()
        
    def update_robot_display(self):
        """更新机械臂显示"""
        # 清除之前的机械臂绘制
        for line in self.ax1.lines[2:]:  # 保留工作空间边界线
            line.remove()
        
        # 获取当前配置的位置
        base, joint1, end_effector = self.robot.get_link_positions(
            self.current_theta1, self.current_theta2)
        
        # 检查碰撞
        collision = False
        for obstacle_center, obstacle_radius in self.obstacles:
            if self.robot.check_collision_with_circle(
                self.current_theta1, self.current_theta2, obstacle_center, obstacle_radius):
                collision = True
                break
        
        # 根据碰撞状态选择颜色
        link_color = 'red' if collision else 'blue'
        joint_color = 'darkred' if collision else 'darkblue'
        
        # 绘制机械臂
        self.ax1.plot([base[0], joint1[0]], [base[1], joint1[1]], 
                     color=link_color, marker='o', linewidth=3, markersize=8, label='Link 1')
        self.ax1.plot([joint1[0], end_effector[0]], [joint1[1], end_effector[1]], 
                     color=joint_color, marker='o', linewidth=3, markersize=8, label='Link 2')
        self.ax1.plot(base[0], base[1], 'ko', markersize=12, label='Base')
        
        # 更新标题显示当前角度和碰撞状态
        status = "Collision" if collision else "No Collision"
        self.ax1.set_title(f'Task Space - θ₁={self.current_theta1:.2f}, θ₂={self.current_theta2:.2f} ({status})')
        
        # 刷新显示
        self.fig.canvas.draw()
        
    def on_click(self, event):
        """处理鼠标点击事件"""
        # 只处理配置空间图上的点击
        if event.inaxes == self.ax2:
            # 获取点击位置的角度值
            self.current_theta1 = event.xdata
            self.current_theta2 = event.ydata
            
            # 确保角度在有效范围内
            if (self.current_theta1 is not None and self.current_theta2 is not None and
                0 <= self.current_theta1 <= 2*np.pi and 0 <= self.current_theta2 <= 2*np.pi):
                
                # 更新配置点标记
                self.config_point.set_data([self.current_theta1], [self.current_theta2])
                
                # 更新机械臂显示
                self.update_robot_display()
                
                # 打印当前配置信息
                print(f"选择的配置: θ₁={self.current_theta1:.3f} rad ({self.current_theta1*180/np.pi:.1f}°), "
                      f"θ₂={self.current_theta2:.3f} rad ({self.current_theta2*180/np.pi:.1f}°)")

def visualize_interactive(robot, obstacles, config_space):
    """
    创建交互式可视化
    """
    viz = InteractiveVisualization(robot, obstacles, config_space)
    plt.tight_layout()
    plt.show()
    return viz

# 主程序示例
if __name__ == "__main__":
    # 创建RR机械臂
    robot = RRRobot(L1=2.0, L2=1.5, theta1_range=[0, 2*np.pi], theta2_range=[0, 2*np.pi])
    
    # 定义障碍物 (圆心坐标, 半径)
    obstacles = [
        ((1.5, 1.0), 0.4),   # 障碍物1
        ((-2.0, 2.0), 0.8),  # 障碍物2
        ((2.5, -1.5), 0.6),  # 障碍物3
    ]
    
    print("正在计算配置空间...")
    
    # 计算配置空间
    config_space = robot.compute_configuration_space(
        obstacles, theta_resolution=100
    )
    
    # 计算可行区域比例
    feasible_ratio = np.sum(config_space) / config_space.size
    print(f"可行配置空间比例: {feasible_ratio:.2%}")
    
    print("启动交互式可视化...")
    print("在右侧配置空间图上点击任意位置来更新左侧机械臂显示")
    
    # 创建交互式可视化
    viz = visualize_interactive(robot, obstacles, config_space)
    
    # 保存配置空间数据
    np.save('config_space.npy', config_space)
    print("配置空间数据已保存到 'config_space.npy'")