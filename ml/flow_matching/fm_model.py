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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Flow Matching + System 2: 利用 Critic 进行偏好选择
#
# 在这个 Notebook 中，我们将演示如何通过引入一个 **Critic** 模型，从 Flow Matching 生成的多模态分布（双峰）中，强行筛选出我们偏好的那一个峰。
#
# ## 目标
# 1. **Actor**: 学习双峰分布（Mode A: (-2, -2), Mode B: (2, 2)）。它会无差别地生成两种样本。
# 2. **Critic**: 学习一个偏好函数，比如我们规定 "Mode B (2, 2) 更好"。
# 3. **System 2 Inference**: 生成大量样本 -> Critic 打分 -> 仅保留高分样本（拒绝采样）。

# %%
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# 修复 macOS 上的 OpenMP 库冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# %% [markdown]
# ## 1. 准备数据：双峰分布
# 这里我们生成两个中心的数据簇：
# - Mode A: (-2, -2)
# - Mode B: (2, 2)

# %%
def generate_bimodal_data(batch_size):
    """
    生成双峰分布数据：
    一半数据在 (-2, -2) 附近 (Mode A)，一半数据在 (2, 2) 附近 (Mode B)。
    """
    n1 = batch_size // 2
    n2 = batch_size - n1
    
    # 峰 A: 中心 (-2, -2)
    data1 = torch.randn(n1, 2) * 0.5 + torch.tensor([-2.0, -2.0])
    
    # 峰 B: 中心 (2, 2)
    data2 = torch.randn(n2, 2) * 0.5 + torch.tensor([2.0, 2.0])
    
    data = torch.cat([data1, data2], dim=0)
    idx = torch.randperm(batch_size)
    return data[idx]

# 可视化
real_data = generate_bimodal_data(1000)
plt.figure(figsize=(5, 5))
plt.scatter(real_data[:, 0], real_data[:, 1], alpha=0.5, s=10, label='Real Data')
plt.title("Target Bimodal Distribution")
plt.legend()
plt.grid(True)
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.show()

# %% [markdown]
# ## 2. 定义 Actor 和 Critic
# - **Actor**: Time-MLP，用于生成。
# - **Critic**: 普通 MLP，输入动作 $x$，输出价值 $V(x)$。

# %%
class TimeMLP(nn.Module):
    """
    Actor: 预测速度场
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.time_emb = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.net = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, t):
        t_emb = self.time_emb(t)
        x_input = torch.cat([x, t_emb], dim=-1)
        return self.net(x_input)

class Critic(nn.Module):
    """
    Critic: 给动作打分。
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # 输出 Value
        )
    
    def forward(self, x):
        return self.net(x)

actor = TimeMLP(input_dim=2, hidden_dim=128, output_dim=2)
critic = Critic(input_dim=2, hidden_dim=64)
print("Models created.")

# %% [markdown]
# ## 3. 训练 Actor (Flow Matching)
# 先让 Actor 学会生成完美的双峰分布。

# %%
def compute_flow_matching_loss(actor, real_action_batch):
    batch_size = real_action_batch.shape[0]
    t = torch.rand(batch_size, 1)
    noise = torch.randn_like(real_action_batch)
    x_t = (1 - t) * noise + t * real_action_batch
    target_velocity = real_action_batch - noise
    pred_velocity = actor(x_t, t)
    loss = nn.MSELoss()(pred_velocity, target_velocity)
    return loss

optimizer_actor = torch.optim.Adam(actor.parameters(), lr=1e-3)
print("Start Training Actor...")
for step in range(2000):
    batch_data = generate_bimodal_data(256)
    optimizer_actor.zero_grad()
    loss = compute_flow_matching_loss(actor, batch_data)
    loss.backward()
    optimizer_actor.step()
    if step % 200 == 0:
        print(f"Step {step}, Loss: {loss.item():.6f}")

# %% [markdown]
# ## 4. 训练 Critic (偏好学习)
# 这里我们模拟一个偏好：我们**只想要右上角 (2, 2) 的动作**。
# 我们定义 Reward 函数为负的欧氏距离：$R(x) = -||x - target||$。

# %%
def reward_function(action):
    # 目标：(2, 2)
    target = torch.tensor([2.0, 2.0])
    # 计算距离
    dist = torch.norm(action - target, dim=1, keepdim=True)
    # 距离越小，Reward 越大
    return -dist

optimizer_critic = torch.optim.Adam(critic.parameters(), lr=1e-3)
print("Start Training Critic...")

for step in range(500):
    # 使用真实数据训练 Critic
    real_actions = generate_bimodal_data(256)
    rewards = reward_function(real_actions)
    
    pred_values = critic(real_actions)
    loss_critic = nn.MSELoss()(pred_values, rewards)
    
    optimizer_critic.zero_grad()
    loss_critic.backward()
    optimizer_critic.step()
    
    if step % 100 == 0:
        print(f"Step {step}, Loss: {loss_critic.item():.6f}")

# %% [markdown]
# ## 5. System 2 推理：拒绝采样
# 1. Actor 生成一批混杂的候选（有的在左下，有的在右上）。
# 2. Critic 给它们打分。
# 3. 我们只取 Top-K 的结果。

# %%
@torch.no_grad()
def solve_ode_inference(actor, batch_size=1, steps=10):
    x_t = torch.randn(batch_size, 2)
    dt = 1.0 / steps
    traj = [x_t.clone()]
    for i in range(steps):
        t_now = torch.ones(batch_size, 1) * (i / steps)
        velocity = actor(x_t, t_now)
        x_t = x_t + velocity * dt
        traj.append(x_t.clone())
    return x_t, traj

@torch.no_grad()
def system_2_inference(actor, critic, num_samples=200, top_k=50):
    # 1. 生成候选
    candidates, traj = solve_ode_inference(actor, batch_size=num_samples, steps=20)
    
    # 2. 打分
    scores = critic(candidates)
    
    # 3. 排序并筛选
    sorted_scores, sorted_indices = torch.sort(scores, dim=0, descending=True)
    top_indices = sorted_indices[:top_k].squeeze()
    best_actions = candidates[top_indices]
    
    return best_actions, candidates, scores, traj

print("Running System 2 Inference...")
best_actions, all_candidates, all_scores, trajectory = system_2_inference(actor, critic, num_samples=200, top_k=50)

# %% [markdown]
# ## 6. 结果可视化
# 左图：Actor 原始生成结果，颜色代表 Critic 打分（黄色高分，紫色低分）。
# 中图：筛选后的 Top-K 结果（应该集中在右上角）。
# 右图：Critic 学到的 Value 分布图。

# %%
plt.figure(figsize=(15, 5))

# 1. 原始生成 (带打分颜色)
plt.subplot(1, 3, 1)
plt.scatter(all_candidates[:, 0], all_candidates[:, 1], c=all_scores.squeeze(), cmap='viridis', alpha=0.6, s=20)
plt.colorbar(label='Critic Value')
plt.title("Raw Actor Output (System 1)")
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.grid(True)

# 2. 筛选后结果
plt.subplot(1, 3, 2)
plt.scatter(best_actions[:, 0], best_actions[:, 1], color='red', alpha=0.6, s=20, label='Selected')
plt.title(f"System 2 Output (Top-K)")
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.grid(True)
plt.legend()

# 3. Critic Landscape
plt.subplot(1, 3, 3)
x = np.linspace(-4, 4, 50)
y = np.linspace(-4, 4, 50)
X, Y = np.meshgrid(x, y)
grid_input = torch.tensor(np.c_[X.ravel(), Y.ravel()], dtype=torch.float32)
with torch.no_grad():
    Z = critic(grid_input).reshape(50, 50)
plt.contourf(X, Y, Z, levels=20, cmap='viridis')
plt.colorbar(label='Predicted Value')
plt.title("Critic Value Landscape")
plt.scatter([2], [2], marker='*', color='red', s=200, label='Target')
plt.legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. 轨迹可视化
# 展示粒子是如何从噪声分布移动到目标分布的。

# %%
# 转换轨迹格式
traj_np = torch.stack(trajectory).numpy() # [steps+1, batch, 2]

plt.figure(figsize=(6, 6))
# 随机选 50 条轨迹绘制
for i in range(50):
    plt.plot(traj_np[:, i, 0], traj_np[:, i, 1], alpha=0.3, color='black', linewidth=0.5)
    
plt.scatter(traj_np[0, :50, 0], traj_np[0, :50, 1], color='red', s=20, label='Start (Noise)')
plt.scatter(traj_np[-1, :50, 0], traj_np[-1, :50, 1], color='blue', s=20, label='End (Generated)')
plt.title("Inference Trajectories (ODE Flow)")
plt.legend()
plt.grid(True)
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.show()
