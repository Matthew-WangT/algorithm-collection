# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: pytorch
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 编写简单的 pi0.6 模型：Flow Matching 拟合双峰分布
# 我们要真正理解 $\pi^*$，最好的办法就是亲手写一个**Mini版**。
# 本示例将构建一个 Flow Matching 模型，尝试学习并生成一个**2D 双峰分布**数据。
#
# ## 核心组件
# 1. **Dataset**: 人工生成的双峰高斯分布（模拟多模态动作空间，如“向左走”或“向右走”）。
# 2. **Actor**: Time-Conditioned MLP，预测速度场 $v_t(x|x_1) = x_1 - x_0$。
# 3. **Solver**: 欧拉积分器，从噪声 $x_0$ 逐步推演到 $x_1$。

# %%
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# 修复 macOS 上的 OpenMP 库冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# %% [markdown]
# ## 0. 准备数据：双峰分布 (Bimodal Distribution)
# 我们需要模拟一个多模态的场景。比如机器人面前有一个障碍物，它可以选择**从左绕行**或**从右绕行**，但不能直接撞上去。
# 这里我们生成两个高斯分布的混合体。

# %%
def generate_bimodal_data(batch_size):
    """
    生成双峰分布数据：
    一半数据在 (-2, -2) 附近，一半数据在 (2, 2) 附近。
    每个峰都有一些高斯噪声。
    """
    n1 = batch_size // 2
    n2 = batch_size - n1
    
    # 峰 1: 中心 (-2, -2)
    data1 = torch.randn(n1, 2) * 0.5 + torch.tensor([-2.0, -2.0])
    
    # 峰 2: 中心 (2, 2)
    data2 = torch.randn(n2, 2) * 0.5 + torch.tensor([2.0, 2.0])
    
    data = torch.cat([data1, data2], dim=0)
    # 打乱顺序
    idx = torch.randperm(batch_size)
    return data[idx]

# 可视化真实数据分布
real_data = generate_bimodal_data(1000)
plt.figure(figsize=(6, 6))
plt.scatter(real_data[:, 0], real_data[:, 1], alpha=0.5, s=10, label='Real Data')
plt.title("Target Bimodal Distribution")
plt.legend()
plt.grid(True)
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.show()


# %% [markdown]
# ## 1. 定义模型 (Actor)
# Flow Matching 的核心是一个 Condition on 时间 $t$ 的 MLP。它输入当前状态 $x_t$ 和时间 $t$，输出向量场（速度）$v$。

# %%
class TimeMLP(nn.Module):
    """
    这是一个通用的 MLP，可以作为 Actor (预测速度场)。
    关键点：必须把时间 t 作为一个输入嵌入进去。
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # 时间嵌入：把标量 t 变成向量，类似 Transformer 的 Positional Encoding
        self.time_emb = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # 主网络
        self.net = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, t):
        # x: 当前状态 (Batch, Dim)
        # t: 当前时间步 (Batch, 1), 范围 [0, 1]
        t_emb = self.time_emb(t)
        # 将状态和时间特征拼接
        x_input = torch.cat([x, t_emb], dim=-1)
        return self.net(x_input)

# 实例化 Actor
actor = TimeMLP(input_dim=2, hidden_dim=128, output_dim=2)
print("Actor model created.")


# %% [markdown]
# ## 2. 训练 Flow Matching
# Flow Matching (FM) 的训练目标非常直观：
# 我们希望构建一个概率流，将噪声分布 $p_0$ (标准正态) 变换为数据分布 $p_1$ (双峰)。
#
# Conditional Flow Matching (CFM) 提出了一种简单的构造方式：
# 对于每一对样本 $(x_0, x_1)$，我们定义一条直线路径 $x_t = (1-t)x_0 + t x_1$。
# 这条路径的速度向量恒定为 $u_t(x|x_1) = x_1 - x_0$。
#
# 我们的模型 $v_\theta(x, t)$ 只需要去回归这个目标速度。

# %%
def compute_flow_matching_loss(actor, real_action_batch):
    batch_size = real_action_batch.shape[0]
    
    # 1. 采样时间 t (0 到 1 之间均匀分布)
    t = torch.rand(batch_size, 1)
    
    # 2. 生成噪声 x_0 (标准正态分布)
    noise = torch.randn_like(real_action_batch)
    
    # 3. 构造中间状态 x_t (线性插值)
    # x_t = (1 - t) * noise + t * real_action
    x_t = (1 - t) * noise + t * real_action_batch
    
    # 4. 计算目标速度 (Target Velocity)
    # 这里的向量场直接指向目标：终点 - 起点
    target_velocity = real_action_batch - noise
    
    # 5. 模型预测
    pred_velocity = actor(x_t, t)
    
    # 6. Loss: MSE
    loss = nn.MSELoss()(pred_velocity, target_velocity)
    return loss

# 开始训练
optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
num_steps = 2000
batch_size = 256

print("Start Training Flow Matching Actor...")
loss_history = []

for step in range(num_steps):
    # 生成真实数据 batch
    batch_data = generate_bimodal_data(batch_size)
    
    optimizer.zero_grad()
    loss = compute_flow_matching_loss(actor, batch_data)
    loss.backward()
    optimizer.step()
    
    loss_history.append(loss.item())
    
    if step % 200 == 0:
        print(f"Step {step}, Loss: {loss.item():.6f}")

plt.plot(loss_history)
plt.title("Training Loss")
plt.xlabel("Step")
plt.ylabel("MSE Loss")
plt.show()


# %% [markdown]
# ## 3. 推理与可视化
# 现在我们使用欧拉积分法 (Euler Method) 来求解 ODE。
# 我们从标准高斯噪声采样 $x_0$，然后让模型 $v_\theta$ 告诉我们该往哪里走，逐步迭代到 $x_1$。

# %%
@torch.no_grad()
def solve_ode_inference(actor, batch_size=1, steps=10):
    """
    欧拉积分法求解 ODE
    """
    # 1. 从纯噪声开始 x_0 ~ N(0, I)
    x_t = torch.randn(batch_size, 2)
    
    # 2. 定义步长 dt
    dt = 1.0 / steps
    
    traj = [x_t.clone()] # 记录轨迹
    
    # 3. 时间循环 0 -> 1
    for i in range(steps):
        # 当前时间 t
        t_now = torch.ones(batch_size, 1) * (i / steps)
        
        # 预测速度
        velocity = actor(x_t, t_now)
        
        # 更新位置
        x_t = x_t + velocity * dt
        
        traj.append(x_t.clone())
        
    return x_t, traj

# 生成样本
print("Generating samples...")
generated_samples, trajectory = solve_ode_inference(actor, batch_size=1000, steps=20)

# 转换轨迹格式
trajectory = torch.stack(trajectory).numpy()

# === 可视化 ===
plt.figure(figsize=(12, 6))

# 左图：最终生成分布
plt.subplot(1, 2, 1)
plt.scatter(real_data[:, 0], real_data[:, 1], alpha=0.1, s=10, label='Real Data (Target)', color='gray')
plt.scatter(generated_samples[:, 0], generated_samples[:, 1], alpha=0.5, s=10, label='Generated', color='blue')
plt.title("Flow Matching Generation Result")
plt.legend()
plt.grid(True)
plt.xlim(-5, 5)
plt.ylim(-5, 5)

# 右图：采样轨迹
plt.subplot(1, 2, 2)
# 随机选 50 条轨迹绘制
for i in range(50):
    plt.plot(trajectory[:, i, 0], trajectory[:, i, 1], alpha=0.3, color='black', linewidth=0.5)
    
plt.scatter(trajectory[0, :50, 0], trajectory[0, :50, 1], color='red', s=20, label='Start (Noise)')
plt.scatter(trajectory[-1, :50, 0], trajectory[-1, :50, 1], color='blue', s=20, label='End (Generated)')
plt.title("Inference Trajectories (ODE Flow)")
plt.legend()
plt.grid(True)
plt.xlim(-5, 5)
plt.ylim(-5, 5)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 结论
# 观察上面的结果，你应该能看到 Flow Matching 的神奇之处：
# 1. **多模态生成**：虽然我们在 Loss 中计算的是 MSE（均方误差），模型并没有简单地输出两个峰的“平均值”（即 (0,0)），而是成功地学会了**分裂**，一部分噪声流向左下角，一部分流向右上角。
# 2. **轨迹平滑**：右图展示了粒子是如何从无序的噪声状态，沿着平滑的路径移动到目标分布的。这就是 ODE Flow 的直观体现。
