# 踝关节机器人MuJoCo仿真控制器

基于MuJoCo物理引擎的踝关节机器人仿真控制系统，支持位置、速度和扭矩三种控制模式。

## 功能特点

- **多种控制模式**: 位置控制、速度控制、扭矩控制
- **PID控制器**: 可调节的PID参数，适用于位置和速度控制
- **实时仿真**: 支持实时和非实时仿真模式
- **可视化渲染**: 集成MuJoCo可视化界面
- **状态监控**: 实时获取关节位置、速度、扭矩信息
- **交互式控制**: 支持键盘交互控制

## 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 基础使用

```python
from ankle_robot import AnkleRobotController

# 创建控制器
controller = AnkleRobotController(xml_path="assets/ankle.xml")

# 位置控制
controller.set_control_mode('position')
controller.set_target_positions([0.2, 0.1, 0.3, -0.3])  # 弧度
controller.run_simulation(duration=5.0)

controller.close()
```

### 运行示例

```bash
# 运行主要演示程序
python ankle_robot.py

# 运行详细使用示例
python example_usage.py
```

## API接口

### 初始化

```python
controller = AnkleRobotController(
    xml_path="assets/ankle.xml",  # XML模型文件路径
    render=True                   # 是否启用可视化
)
```

### 控制模式

```python
# 设置控制模式
controller.set_control_mode('position')  # 'position', 'velocity', 'torque'

# 位置控制
controller.set_target_positions([ankle_pitch, ankle_roll, ankle_l_joint, ankle_r_joint])

# 速度控制
controller.set_target_velocities([vel1, vel2, vel3, vel4])  # 弧度/秒

# 扭矩控制
controller.set_target_torques([torque1, torque2, torque3, torque4])  # N·m
```

### PID参数调节

```python
controller.set_position_control_gains(
    kp=[100.0, 100.0, 50.0, 50.0],  # 比例增益
    ki=[1.0, 1.0, 0.5, 0.5],        # 积分增益
    kd=[10.0, 10.0, 5.0, 5.0]       # 微分增益
)
```

### 状态获取

```python
# 获取关节位置
positions = controller.get_joint_positions()

# 获取关节速度
velocities = controller.get_joint_velocities()

# 获取关节扭矩
torques = controller.get_joint_torques()

# 获取完整状态信息
state = controller.get_state_info()
```

### 仿真控制

```python
# 单步仿真
controller.step()

# 运行仿真
controller.run_simulation(duration=10.0, real_time=True)

# 重置仿真
controller.reset()

# 停止仿真
controller.stop_simulation()
```

## 关节配置

踝关节机器人包含4个关节：

1. **ankle_pitch**: 踝关节俯仰运动 (绕Y轴)
2. **ankle_roll**: 踝关节翻滚运动 (绕X轴)  
3. **ankle_l_joint**: 左侧连杆关节
4. **ankle_r_joint**: 右侧连杆关节

## 控制模式详解

### 位置控制
- 使用PID控制器跟踪目标位置
- 适用于精确定位任务
- 可调节PID参数以优化响应

### 速度控制
- 基于比例控制的速度跟踪
- 适用于匀速运动任务
- 可设置目标角速度

### 扭矩控制
- 直接设置关节扭矩输出
- 适用于力控制和动力学仿真
- 可模拟外力干扰

## 交互式控制

运行主程序后，可使用键盘进行交互式控制：

- `w/s`: ankle_pitch +/-
- `a/d`: ankle_roll +/-
- `i/k`: ankle_l_joint +/-
- `j/l`: ankle_r_joint +/-
- `r`: 重置位置
- `q`: 退出

## 注意事项

1. 确保`ankle.xml`文件在`assets/`目录下
2. 首次运行可能需要下载MuJoCo模型文件
3. 实时仿真需要足够的计算性能
4. 关节限位范围已在XML文件中定义
5. PID参数需要根据具体应用调节

## 故障排除

### 常见问题

1. **模型文件找不到**: 检查XML文件路径是否正确
2. **仿真运行缓慢**: 关闭可视化渲染或降低仿真精度
3. **控制不稳定**: 调节PID参数，特别是积分增益
4. **关节超限**: 检查目标值是否在关节限位范围内

### 调试模式

```python
# 启用详细状态输出
state = controller.get_state_info()
print(f"当前状态: {state}")

# 监控控制输出
controller.update_control()
print(f"控制输出: {controller.data.ctrl}")
```

## 扩展开发

可以基于`AnkleRobotController`类进行二次开发：

- 添加新的控制算法
- 集成传感器反馈
- 实现轨迹规划
- 添加安全保护机制
