import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import matplotlib.patches as patches

from rr_robot import RRRobot
from AStar import AStarPlanner


class InteractiveVisualization:
    def __init__(self, robot, obstacles, config_space):
        self.robot = robot
        self.obstacles = obstacles
        self.config_space = config_space
        self.current_theta1 = np.pi/4
        self.current_theta2 = np.pi/4
        
        # 路径规划相关
        self.planner = AStarPlanner(config_space)
        self.start_point = None
        self.goal_point = None
        self.planned_path = None
        self.mode = 'navigate'  # 'navigate', 'set_start', 'set_goal'
        
        # 动画相关
        self.is_playing = False
        self.animation_timer = None
        self.current_path_index = 0
        self.animation_speed = 200  # 毫秒
        
        # 创建图形
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 初始化任务空间图
        self.setup_task_space()
        
        # 初始化配置空间图
        self.setup_config_space()
        
        # 绘制初始机械臂配置
        self.update_robot_display()
        
        # 连接事件
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # 添加控制说明
        self.add_instructions()
        
    def add_instructions(self):
        """添加控制说明"""
        instruction_text = (
            "Control Instructions:\n"
            "Click on configuration space: Move robot\n"
            "Press 's': Set start point mode\n"
            "Press 'g': Set goal point mode\n"
            "Press 'p': Plan path and play\n"
            "Press 'c': Clear path\n"
            "Press 'n': Return to navigation mode\n"
            "Press 'space': Pause/Resume playback"
        )
        self.fig.text(0.02, 0.98, instruction_text, transform=self.fig.transFigure, 
                     fontsize=10, verticalalignment='top', 
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        
    def setup_task_space(self):
        """设置任务空间图"""
        self.ax1.set_xlim(-self.robot.L1-self.robot.L2-1, self.robot.L1+self.robot.L2+1)
        self.ax1.set_ylim(-self.robot.L1-self.robot.L2-1, self.robot.L1+self.robot.L2+1)
        self.ax1.set_aspect('equal')
        self.ax1.set_title('Task Space')
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
        self.ax2.set_title('Configuration Space - Navigation Mode')
        
        # 添加刻度标签
        self.ax2.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        self.ax2.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
        self.ax2.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        self.ax2.set_yticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=self.ax2)
        cbar.set_label('Feasibility (1=Feasible, 0=Infeasible)')
        
        # 初始化标记点
        self.config_point, = self.ax2.plot(self.current_theta1, self.current_theta2, 
                                          'bo', markersize=10, label='Current Configuration')
        self.start_marker, = self.ax2.plot([], [], 'go', markersize=12, label='Start Point')
        self.goal_marker, = self.ax2.plot([], [], 'ro', markersize=12, label='Goal Point')
        self.path_line, = self.ax2.plot([], [], 'b-', linewidth=2, alpha=0.7, label='Planned Path')
        
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
        
    def on_key_press(self, event):
        """处理键盘事件"""
        if event.key == 's':
            self.mode = 'set_start'
            self.ax2.set_title('Configuration Space - Set Start Mode (Click to select start point)')
            print("Enter set start mode, click on configuration space to select start point")
        elif event.key == 'g':
            self.mode = 'set_goal'
            self.ax2.set_title('Configuration Space - Set Goal Mode (Click to select goal point)')
            print("Enter set goal mode, click on configuration space to select goal point")
        elif event.key == 'n':
            self.mode = 'navigate'
            self.ax2.set_title('Configuration Space - Navigation Mode')
            print("Return to navigation mode")
        elif event.key == 'p':
            self.plan_path()
        elif event.key == 'c':
            self.clear_path()
        elif event.key == ' ':  # 空格键暂停/继续
            self.toggle_animation()
        
        self.fig.canvas.draw()
        
    def on_click(self, event):
        """处理鼠标点击事件"""
        # 只处理配置空间图上的点击
        if event.inaxes == self.ax2:
            # 获取点击位置的角度值
            theta1 = event.xdata
            theta2 = event.ydata
            
            # 确保角度在有效范围内
            if (theta1 is not None and theta2 is not None and
                0 <= theta1 <= 2*np.pi and 0 <= theta2 <= 2*np.pi):
                
                if self.mode == 'navigate':
                    # 导航模式：更新当前配置
                    self.current_theta1 = theta1
                    self.current_theta2 = theta2
                    self.config_point.set_data([self.current_theta1], [self.current_theta2])
                    self.update_robot_display()
                    print(f"Current configuration: θ₁={self.current_theta1:.3f} rad ({self.current_theta1*180/np.pi:.1f}°), "
                          f"θ₂={self.current_theta2:.3f} rad ({self.current_theta2*180/np.pi:.1f}°)")
                    
                elif self.mode == 'set_start':
                    # 设置起点
                    self.start_point = (theta1, theta2)
                    self.start_marker.set_data([theta1], [theta2])
                    print(f"Start point set to: θ₁={theta1:.3f} rad, θ₂={theta2:.3f} rad")
                    self.mode = 'navigate'
                    self.ax2.set_title('Configuration Space - Navigation Mode')
                    
                elif self.mode == 'set_goal':
                    # 设置终点
                    self.goal_point = (theta1, theta2)
                    self.goal_marker.set_data([theta1], [theta2])
                    print(f"Goal point set to: θ₁={theta1:.3f} rad, θ₂={theta2:.3f} rad")
                    self.mode = 'navigate'
                    self.ax2.set_title('Configuration Space - Navigation Mode')
                
                self.fig.canvas.draw()
                
    def plan_path(self):
        """规划路径"""
        if self.start_point is None:
            print("Please set start point first (press 's' key)")
            return
        if self.goal_point is None:
            print("Please set goal point first (press 'g' key)")
            return
            
        print("Planning path...")
        self.planned_path = self.planner.plan_path(self.start_point, self.goal_point)
        
        if self.planned_path:
            print(f"Path planning successful! Path length: {len(self.planned_path)} configuration points")
            
            # 绘制路径
            path_theta1 = [config[0] for config in self.planned_path]
            path_theta2 = [config[1] for config in self.planned_path]
            self.path_line.set_data(path_theta1, path_theta2)
            
            # 添加路径点标记
            self.ax2.plot(path_theta1, path_theta2, 'b.', markersize=4, alpha=0.6)
            
            self.fig.canvas.draw()
            
            # 打印路径信息
            print("Path details:")
            for i, config in enumerate(self.planned_path):
                print(f"  Step {i}: θ₁={config[0]:.3f}, θ₂={config[1]:.3f}")
                
            # 立即开始播放轨迹
            print("Starting trajectory animation...")
            self.start_animation()
        else:
            print("Path planning failed!")
            
    def start_animation(self):
        """开始播放轨迹动画"""
        if self.planned_path is None or len(self.planned_path) == 0:
            return
            
        self.is_playing = True
        self.current_path_index = 0
        
        # 停止之前的动画（如果有的话）
        if self.animation_timer is not None:
            self.animation_timer.stop()
            
        # 创建新的动画定时器
        self.animation_timer = self.fig.canvas.new_timer(interval=self.animation_speed)
        self.animation_timer.add_callback(self.animate_step)
        self.animation_timer.start()
        
    def stop_animation(self):
        """停止动画播放"""
        self.is_playing = False
        if self.animation_timer is not None:
            self.animation_timer.stop()
            self.animation_timer = None
            
    def animate_step(self):
        """动画的单步执行"""
        if not self.is_playing or self.planned_path is None:
            return
            
        if self.current_path_index < len(self.planned_path):
            # 更新当前配置
            config = self.planned_path[self.current_path_index]
            self.current_theta1, self.current_theta2 = config
            
            # 更新配置空间中的当前位置标记
            self.config_point.set_data([self.current_theta1], [self.current_theta2])
            
            # 更新机械臂显示
            self.update_robot_display()
            
            # 在标题中显示进度
            progress = f"Progress: {self.current_path_index + 1}/{len(self.planned_path)}"
            self.ax2.set_title(f'Configuration Space - Playing Trajectory ({progress})')
            
            print(f"Playing step {self.current_path_index + 1}/{len(self.planned_path)}: "
                  f"θ₁={self.current_theta1:.3f}, θ₂={self.current_theta2:.3f}")
            
            self.current_path_index += 1
        else:
            # 动画播放完成
            self.stop_animation()
            print("Trajectory playback completed!")
            self.ax2.set_title('Configuration Space - Navigation Mode')
            
    def toggle_animation(self):
        """切换动画播放状态"""
        if self.planned_path is None:
            print("No path available for playback")
            return
            
        if self.is_playing:
            self.stop_animation()
            print("Animation paused")
            self.ax2.set_title('Configuration Space - Animation Paused (Press space to continue)')
        else:
            if self.current_path_index >= len(self.planned_path):
                # 如果已经播放完成，重新开始
                self.current_path_index = 0
            self.start_animation()
            print("Animation resumed")
            
    def clear_path(self):
        """清除路径"""
        # 停止动画
        self.stop_animation()
        
        self.planned_path = None
        self.path_line.set_data([], [])
        
        # 清除路径点标记
        for line in self.ax2.lines:
            if line.get_marker() == '.' and line.get_color() == 'b':
                line.remove()
        
        self.fig.canvas.draw()
        print("Path cleared")

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
    
    print("Computing configuration space...")
    
    # 计算配置空间
    config_space = robot.compute_configuration_space(
        obstacles, theta_resolution=100
    )
    
    # 计算可行区域比例
    feasible_ratio = np.sum(config_space) / config_space.size
    print(f"Feasible configuration space ratio: {feasible_ratio:.2%}")
    
    print("Starting interactive visualization...")
    print("Click anywhere on the right configuration space plot to update the robot display on the left")
    
    # 创建交互式可视化
    viz = visualize_interactive(robot, obstacles, config_space)
