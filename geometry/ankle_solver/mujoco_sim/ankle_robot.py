import numpy as np
import mujoco
import mujoco.viewer
import time
import threading
from typing import List, Optional, Dict, Any


class AnkleRobotController:
    """
    踝关节机器人MuJoCo仿真控制器
    支持位置控制、速度控制和扭矩控制
    """
    
    def __init__(self, xml_path: str = "assets/ankle.xml", render: bool = True):
        """
        初始化踝关节机器人控制器
        
        Args:
            xml_path: XML模型文件路径
            render: 是否启用可视化渲染
        """
        self.xml_path = xml_path
        self.render_enabled = render
        
        # 加载模型
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # 关节信息
        self.joint_names = ['ankle_pitch', 'ankle_roll', 'ankle_l_joint', 'ankle_r_joint']
        self.actuator_names = ['ankle_pitch_motor', 'ankle_roll_motor', 'ankle_l_motor', 'ankle_r_motor']
        
        # 获取关节和电机索引
        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) 
                         for name in self.joint_names]
        self.actuator_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) 
                            for name in self.actuator_names]
        
        # 控制模式
        self.control_mode = 'position'  # 'position', 'velocity', 'torque'
        
        # PID控制器参数（用于位置和速度控制）
        self.kp = 1.0*np.array([1.0, 1.0, 0.0, 0.0])  # 比例增益
        self.ki = 0.0*np.array([1.0, 1.0, 0.0, 0.0])  # 积分增益
        self.kd = 0.1*np.array([1.0, 1.0, 0.0, 0.0])  # 微分增益
        
        # PID控制器状态
        self.integral_error = np.zeros(4)
        self.prev_error = np.zeros(4)
        
        # 目标值
        self.target_positions = np.zeros(4)
        self.target_velocities = np.zeros(4)
        self.target_torques = np.zeros(4)
        
        # 仿真参数
        self.dt = self.model.opt.timestep
        self.simulation_running = False
        
        # 渲染器
        self.viewer = None
        if self.render_enabled:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        
        print(f"踝关节机器人控制器初始化完成")
        print(f"关节数量: {len(self.joint_names)}")
        print(f"关节名称: {self.joint_names}")
        print(f"时间步长: {self.dt:.4f}s")
    
    def get_joint_positions(self) -> np.ndarray:
        """获取当前关节位置"""
        positions = np.zeros(4)
        for i, joint_id in enumerate(self.joint_ids):
            positions[i] = self.data.qpos[joint_id]
        return positions
    
    def get_joint_velocities(self) -> np.ndarray:
        """获取当前关节速度"""
        velocities = np.zeros(4)
        for i, joint_id in enumerate(self.joint_ids):
            velocities[i] = self.data.qvel[joint_id]
        return velocities
    
    def get_joint_torques(self) -> np.ndarray:
        """获取当前关节扭矩"""
        torques = np.zeros(4)
        for i, actuator_id in enumerate(self.actuator_ids):
            torques[i] = self.data.actuator_force[actuator_id]
        return torques
    
    def set_position_control_gains(self, kp: List[float], ki: List[float], kd: List[float]):
        """设置位置控制PID参数"""
        self.kp = np.array(kp)
        self.ki = np.array(ki)
        self.kd = np.array(kd)
        print(f"位置控制增益已更新: Kp={self.kp}, Ki={self.ki}, Kd={self.kd}")
    
    def set_control_mode(self, mode: str):
        """
        设置控制模式
        
        Args:
            mode: 'position', 'velocity', 'torque'
        """
        if mode not in ['position', 'velocity', 'torque']:
            raise ValueError("控制模式必须是 'position', 'velocity', 或 'torque'")
        
        self.control_mode = mode
        # 重置PID状态
        self.integral_error = np.zeros(4)
        self.prev_error = np.zeros(4)
        
        print(f"控制模式已切换为: {mode}")
    
    def set_target_positions(self, positions: List[float]):
        """
        设置目标关节位置（弧度）
        
        Args:
            positions: [ankle_pitch, ankle_roll, ankle_l_joint, ankle_r_joint]
        """
        if len(positions) != 4:
            raise ValueError("必须提供4个关节的目标位置")
        
        self.target_positions = np.array(positions)
        print(f"目标位置已设置: {self.target_positions}")
    
    def set_target_velocities(self, velocities: List[float]):
        """
        设置目标关节速度（弧度/秒）
        
        Args:
            velocities: [ankle_pitch, ankle_roll, ankle_l_joint, ankle_r_joint]
        """
        if len(velocities) != 4:
            raise ValueError("必须提供4个关节的目标速度")
        
        self.target_velocities = np.array(velocities)
        print(f"目标速度已设置: {self.target_velocities}")
    
    def set_target_torques(self, torques: List[float]):
        """
        设置目标关节扭矩（N·m）
        
        Args:
            torques: [ankle_pitch, ankle_roll, ankle_l_joint, ankle_r_joint]
        """
        if len(torques) != 4:
            raise ValueError("必须提供4个关节的目标扭矩")
        
        self.target_torques = np.array(torques)
        print(f"目标扭矩已设置: {self.target_torques}")
    
    def _compute_position_control(self) -> np.ndarray:
        """计算位置控制输出"""
        current_positions = self.get_joint_positions()
        current_velocities = self.get_joint_velocities()
        
        # PID控制
        error = self.target_positions - current_positions
        self.integral_error += error * self.dt
        derivative_error = (error - self.prev_error) / self.dt
        
        control_output = (self.kp * error + 
                         self.ki * self.integral_error + 
                         self.kd * derivative_error)
        
        self.prev_error = error
        
        return control_output
    
    def _compute_velocity_control(self) -> np.ndarray:
        """计算速度控制输出"""
        current_velocities = self.get_joint_velocities()
        
        # 简单的比例控制
        error = self.target_velocities - current_velocities
        control_output = self.kp * error
        
        return control_output
    
    def _compute_torque_control(self) -> np.ndarray:
        """计算扭矩控制输出"""
        return self.target_torques.copy()
    
    def update_control(self):
        """更新控制输出"""
        if self.control_mode == 'position':
            control_output = self._compute_position_control()
        elif self.control_mode == 'velocity':
            control_output = self._compute_velocity_control()
        elif self.control_mode == 'torque':
            control_output = self._compute_torque_control()
        else:
            control_output = np.zeros(4)
        
        # 应用控制输出到电机
        for i, actuator_id in enumerate(self.actuator_ids):
            self.data.ctrl[actuator_id] = control_output[i]
    
    def step(self):
        """执行一步仿真"""
        self.update_control()
        mujoco.mj_step(self.model, self.data)
        
        if self.render_enabled and self.viewer is not None:
            self.viewer.sync()
    
    def run_simulation(self, duration: float = 10.0, real_time: bool = True):
        """
        运行仿真
        
        Args:
            duration: 仿真时长（秒）
            real_time: 是否实时运行
        """
        print(f"开始仿真，时长: {duration}秒")
        
        self.simulation_running = True
        start_time = time.time()
        step_count = 0
        
        try:
            while self.simulation_running and (time.time() - start_time) < duration:
                step_start = time.time()
                
                self.step()
                step_count += 1
                
                # 实时控制
                if real_time:
                    elapsed = time.time() - step_start
                    sleep_time = max(0, self.dt - elapsed)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                
                # 每100步打印一次状态
                if step_count % 100 == 0:
                    pos = self.get_joint_positions()
                    vel = self.get_joint_velocities()
                    print(f"步数: {step_count}, 位置: {pos}, 速度: {vel}")
        
        except KeyboardInterrupt:
            print("仿真被用户中断")
        
        finally:
            self.simulation_running = False
            print(f"仿真结束，总步数: {step_count}")
    
    def stop_simulation(self):
        """停止仿真"""
        self.simulation_running = False
    
    def reset(self):
        """重置仿真状态"""
        mujoco.mj_resetData(self.model, self.data)
        self.integral_error = np.zeros(4)
        self.prev_error = np.zeros(4)
        print("仿真状态已重置")
    
    def get_state_info(self) -> Dict[str, Any]:
        """获取当前状态信息"""
        return {
            'time': self.data.time,
            'positions': self.get_joint_positions().tolist(),
            'velocities': self.get_joint_velocities().tolist(),
            'torques': self.get_joint_torques().tolist(),
            'control_mode': self.control_mode,
            'target_positions': self.target_positions.tolist(),
            'target_velocities': self.target_velocities.tolist(),
            'target_torques': self.target_torques.tolist()
        }
    
    def close(self):
        """关闭仿真器"""
        if self.viewer is not None:
            self.viewer.close()
        print("仿真器已关闭")


def demo_position_control():
    """演示位置控制"""
    print("\n=== 位置控制演示 ===")
    
    controller = AnkleRobotController()
    controller.set_control_mode('position')
    
    # 设置目标位置
    target_pos = [0.2, 0.1, 0.3, -0.3]  # 弧度
    controller.set_target_positions(target_pos)
    
    # 运行仿真
    controller.run_simulation(duration=5.0)
    controller.close()


def demo_velocity_control():
    """演示速度控制"""
    print("\n=== 速度控制演示 ===")
    
    controller = AnkleRobotController()
    controller.set_control_mode('velocity')
    
    # 设置目标速度
    target_vel = [0.5, -0.3, 0.2, 0.1]  # 弧度/秒
    controller.set_target_velocities(target_vel)
    
    # 运行仿真
    controller.run_simulation(duration=3.0)
    controller.close()


def demo_torque_control():
    """演示扭矩控制"""
    print("\n=== 扭矩控制演示 ===")
    
    controller = AnkleRobotController()
    controller.set_control_mode('torque')
    
    # 设置目标扭矩
    target_torque = [0.5, -0.5, 0.0, -0.0]  # N·m
    controller.set_target_torques(target_torque)
    
    # 运行仿真
    controller.run_simulation(duration=3.0)
    controller.close()


def interactive_demo():
    """交互式演示"""
    print("\n=== 交互式演示 ===")
    print("使用WASD键控制踝关节运动")
    print("按'q'退出")
    
    controller = AnkleRobotController()
    controller.set_control_mode('position')
    
    # 初始位置
    current_targets = [0.0, 0.0, 0.0, 0.0]
    step_size = 0.1
    
    def simulation_thread():
        """仿真线程"""
        while controller.simulation_running:
            controller.step()
            time.sleep(controller.dt)
    
    # 启动仿真线程
    controller.simulation_running = True
    sim_thread = threading.Thread(target=simulation_thread)
    sim_thread.daemon = True
    sim_thread.start()
    
    try:
        import sys, select, tty, termios
        
        # 设置终端为非阻塞模式
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.cbreak(sys.stdin.fileno())
        except AttributeError:
            # 某些Python版本使用setcbreak
            tty.setcbreak(sys.stdin.fileno())
        
        print("控制说明:")
        print("w/s: ankle_pitch +/-")
        print("a/d: ankle_roll +/-") 
        print("i/k: ankle_l_joint +/-")
        print("j/l: ankle_r_joint +/-")
        print("r: 重置位置")
        print("q: 退出")
        
        while True:
            if select.select([sys.stdin], [], [], 0.1) == ([sys.stdin], [], []):
                key = sys.stdin.read(1)
                
                if key == 'q':
                    break
                elif key == 'w':
                    current_targets[0] += step_size
                elif key == 's':
                    current_targets[0] -= step_size
                elif key == 'a':
                    current_targets[1] += step_size
                elif key == 'd':
                    current_targets[1] -= step_size
                elif key == 'i':
                    current_targets[2] += step_size
                elif key == 'k':
                    current_targets[2] -= step_size
                elif key == 'j':
                    current_targets[3] += step_size
                elif key == 'l':
                    current_targets[3] -= step_size
                elif key == 'r':
                    current_targets = [0.0, 0.0, 0.0, 0.0]
                
                controller.set_target_positions(current_targets)
                print(f"目标位置: {current_targets}")
    
    except ImportError:
        print("交互式控制需要Unix系统支持")
        # 简单的自动演示
        for i in range(50):
            targets = [0.3 * np.sin(i * 0.1), 
                      0.2 * np.cos(i * 0.1),
                      0.1 * np.sin(i * 0.05),
                      -0.1 * np.cos(i * 0.05)]
            controller.set_target_positions(targets)
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        pass
    
    finally:
        if 'old_settings' in locals():
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        controller.stop_simulation()
        controller.close()


if __name__ == "__main__":
    print("踝关节机器人MuJoCo仿真控制器")
    print("=" * 40)
    
    # 运行各种演示
    # demo_position_control()
    # demo_velocity_control()
    # demo_torque_control()
    interactive_demo()
