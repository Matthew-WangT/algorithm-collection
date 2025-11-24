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
# # Flow Matching + IQL: 从次优演示中学习最优策略
#
# 在这个 Notebook 中，我们将通过一个更复杂的**绕障任务**来演示 IQL (Implicit Q-Learning) 的核心价值。
#
# ## 任务描述
# - **目标**: 从起点 (0, -3) 移动到终点 (0, 3)。
# - **障碍**: 原点 (0, 0) 有一个圆形障碍物 (半径 1)。
# - **数据**: 包含大量**次优轨迹**。有的绕得很远（安全但慢），有的绕得近（快）。我们甚至可以混入一些失败的轨迹。
#
# ## 为什么需要 IQL?
# 如果我们只用标准的 Flow Matching (Behavior Cloning)，模型会学习数据的**平均行为**——也就是绕一个不远不近的圈子。
# 但我们需要模型学会**最优行为**（紧贴障碍物边缘，路径最短）。
# IQL 通过 Expectile Regression，能够从分布中“挑”出那些高价值的样本进行学习，从而在推理时通过 Critic 筛选出优于平均水平的轨迹。

# %%
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# 修复 macOS 上的 OpenMP 库冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# %% [markdown]
# ## 1. 准备数据：有噪声的绕障轨迹
# 我们模拟生成两种绕行策略（左/右），但增加较大的随机性，模拟人类操作时的不完美（绕远路）。

# %%
def generate_obstacle_data(batch_size):
    """
    生成绕过 (0,0) 障碍物的轨迹数据。
    为了简化 Flow Matching 训练，我们这里直接生成目标点分布（一步到位的 Action），
    而不是完整的时序轨迹。
    
    假设这是一个 Contextual Bandit 问题：
    - State: 起点 (固定为 0, -3，或者加点噪声)
    - Action: 中间途经点 (比如在 y=0 截面上的 x 坐标)
    
    为了演示 Flow Matching 的生成能力，我们生成 2D 空间中的点，
    这些点代表了"成功的轨迹"经过的关键路点 (Waypoints)。
    
    这里我们生成两个弯月形的数据簇，代表从左绕和从右绕。
    """
    n = batch_size
    n1 = n // 2
    n2 = n - n1
    
    # 方案 C (最终方案):
    # 左簇: x ~ N(-1.5, 0.5), y ~ N(0, 1)
    # 右簇: x ~ N(1.5, 0.5), y ~ N(0, 1)
    # 奖励: 离 (0,0) 越近 (但 > 1) 奖励越高。
    # 这样数据大部分分布在 x=±1.5 (次优)，最优策略是 x=±1.0
    
    data1 = torch.randn(n1, 2) * torch.tensor([0.5, 1.0]) + torch.tensor([-1.8, 0.0])
    data2 = torch.randn(n2, 2) * torch.tensor([0.5, 1.0]) + torch.tensor([1.8, 0.0])
    
    data = torch.cat([data1, data2], dim=0)
    
    # 强制过滤掉所有距离原点小于 1.1 的点 (避免撞墙)
    dist = torch.norm(data, dim=1)
    valid_mask = dist > 1.1
    data = data[valid_mask]
    
    # 如果过滤后数量不够，简单补齐
    if len(data) < batch_size:
        # 递归补齐比较慢，这里直接用复制补齐
        repeats = (batch_size // len(data)) + 1
        data = data.repeat(repeats, 1)
        data = data[:batch_size]
        
    return data

# 可视化
real_data = generate_obstacle_data(1000)
plt.figure(figsize=(6, 6))
plt.scatter(real_data[:, 0], real_data[:, 1], alpha=0.5, s=10, label='Demonstrations')
circle = plt.Circle((0, 0), 1.0, color='black', alpha=0.3, label='Obstacle')
plt.gca().add_patch(circle)
plt.title("Sub-optimal Demonstrations (Avoid Obstacle)")
plt.legend()
plt.grid(True)
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.show()

# %% [markdown]
# ## 2. 定义 Actor 和 Critic
# 模型结构保持不变。

# %%
class TimeMLP(nn.Module):
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
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # Output Value
        )
    
    def forward(self, x):
        return self.net(x)

actor = TimeMLP(input_dim=2, hidden_dim=128, output_dim=2)
critic = Critic(input_dim=2, hidden_dim=64)

# %% [markdown]
# ## 3. 训练 Actor (Behavior Cloning)
# Actor 会学习数据的分布。由于数据大部分集中在 x=±1.8 (远离障碍)，Actor 生成的样本也会倾向于绕远路。

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
    batch_data = generate_obstacle_data(256)
    optimizer_actor.zero_grad()
    loss = compute_flow_matching_loss(actor, batch_data)
    loss.backward()
    optimizer_actor.step()
    if step % 500 == 0:
        print(f"Step {step}, Loss: {loss.item():.6f}")

# %% [markdown]
# ## 4. 定义 Reward 并训练 Critic
# **关键点**：我们的 Reward 函数鼓励**紧贴障碍物** (r -> 1.0)。
#
# $$ R(x) = -| ||x|| - 1.0 | $$
#
# 即：距离原点越接近 1.0，奖励越高。距离越远（无论是撞墙还是绕太远），奖励越低。
#
# 注意：我们的数据集中，大部分数据的 $||x|| \approx 1.8$，只有极少数噪声数据偶然落在了 $||x|| \approx 1.1$ 附近。
# **IQL 的任务就是从这些极少数的“好运”样本中，学会高价值区域在 r=1 附近。**

# %%
def reward_function(action):
    # 距离原点的距离
    r = torch.norm(action, dim=1, keepdim=True)
    
    # 奖励：越接近 1.1 (安全边界) 越好
    # 惩罚撞墙 (r < 1.0) 和 绕远 (r > 1.5)
    
    # 定义一个更 sharp 的奖励
    # 如果 r < 1.0 (撞墙): -10
    # 否则: 1.0 / (r - 0.9)  (越近越好)
    
    reward = -torch.abs(r - 1.2) * 5.0 # 目标半径 1.2
    
    # 撞墙惩罚
    collision_mask = (r < 1.0).float()
    reward = reward - collision_mask * 10.0
    
    return reward

# %% [markdown]
# ## 5. 对比实验：MSE vs IQL
# 我们训练两个 Critic：
# 1. **MSE Critic**: 学习平均奖励 (Expected Value)。
# 2. **IQL Critic**: 学习最优奖励 (Expectile Value, $\tau=0.95$)。
#
# **重要修改**: 为了防止 Critic 对障碍物内部产生错误的高估（因为它没见过里面的坏数据），我们引入**Negative Mining**。在训练时混入障碍物内部的随机点作为负样本。

# %%
critic_mse = Critic(input_dim=2, hidden_dim=64)
critic_iql = Critic(input_dim=2, hidden_dim=64)

opt_mse = torch.optim.Adam(critic_mse.parameters(), lr=1e-3)
opt_iql = torch.optim.Adam(critic_iql.parameters(), lr=1e-3)

def iql_expectile_loss(pred, target, expectile=0.95):
    diff = target - pred
    weight = torch.where(diff > 0, expectile, (1 - expectile))
    return torch.mean(weight * (diff ** 2))

print("Start Training Critics with Negative Mining...")
for step in range(2000):
    # 1. 真实数据 (Positive Samples)
    real_actions = generate_obstacle_data(128)
    rewards_real = reward_function(real_actions)
    
    # 2. 负样本 (Negative Mining)
    # 重点在障碍物内部采样：r ~ U[0, 1.1]
    r_neg = torch.rand(128, 1) * 1.1
    theta_neg = torch.rand(128, 1) * 2 * np.pi
    neg_actions = torch.cat([r_neg * torch.cos(theta_neg), r_neg * torch.sin(theta_neg)], dim=1)
    rewards_neg = reward_function(neg_actions) # 自动给出低分
    
    # 合并
    combined_actions = torch.cat([real_actions, neg_actions], dim=0)
    combined_rewards = torch.cat([rewards_real, rewards_neg], dim=0)
    
    # 1. MSE Update
    pred_mse = critic_mse(combined_actions)
    loss_mse = nn.MSELoss()(pred_mse, combined_rewards)
    opt_mse.zero_grad()
    loss_mse.backward()
    opt_mse.step()
    
    # 2. IQL Update
    pred_iql = critic_iql(combined_actions)
    loss_iql = iql_expectile_loss(pred_iql, combined_rewards, expectile=0.95)
    opt_iql.zero_grad()
    loss_iql.backward()
    opt_iql.step()
    
    if step % 500 == 0:
        print(f"Step {step} | MSE Loss: {loss_mse.item():.4f} | IQL Loss: {loss_iql.item():.4f}")

# %% [markdown]
# ## 6. System 2 推理与可视化
# 我们对比三种策略：
# 1. **Raw Actor**: 原始行为克隆（应该绕得远）。
# 2. **MSE Selection**: 用普通 Critic 筛选（可能提升不大）。
# 3. **IQL Selection**: 用 IQL Critic 筛选（应该能选出紧贴障碍的样本）。

# %%
@torch.no_grad()
def solve_ode_inference(actor, batch_size=1, steps=10):
    x_t = torch.randn(batch_size, 2)
    dt = 1.0 / steps
    for i in range(steps):
        t_now = torch.ones(batch_size, 1) * (i / steps)
        velocity = actor(x_t, t_now)
        x_t = x_t + velocity * dt
    return x_t

# 生成大量候选
candidates = solve_ode_inference(actor, batch_size=1000, steps=20)

# 计算不同 Critic 的打分
scores_mse = critic_mse(candidates)
scores_iql = critic_iql(candidates)

# 筛选 Top-50
top_k = 50
_, idx_mse = torch.sort(scores_mse, dim=0, descending=True)
best_mse = candidates[idx_mse[:top_k].squeeze()]

_, idx_iql = torch.sort(scores_iql, dim=0, descending=True)
best_iql = candidates[idx_iql[:top_k].squeeze()]

# 可视化
plt.figure(figsize=(18, 6))

# 1. 原始 Actor 分布
plt.subplot(1, 3, 1)
plt.scatter(candidates[:, 0], candidates[:, 1], alpha=0.3, s=10, color='gray', label='Candidates')
circle = plt.Circle((0, 0), 1.0, color='black', alpha=0.3)
plt.gca().add_patch(circle)
plt.title(f"Raw Actor (Avg Dist: {torch.norm(candidates, dim=1).mean():.2f})")
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.grid(True)

# 2. MSE 筛选
plt.subplot(1, 3, 2)
plt.scatter(candidates[:, 0], candidates[:, 1], alpha=0.1, s=10, color='gray')
plt.scatter(best_mse[:, 0], best_mse[:, 1], color='orange', s=30, label='MSE Selected')
circle = plt.Circle((0, 0), 1.0, color='black', alpha=0.3)
plt.gca().add_patch(circle)
plt.title(f"MSE Critic (Avg Dist: {torch.norm(best_mse, dim=1).mean():.2f})")
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.legend()
plt.grid(True)

# 3. IQL 筛选
plt.subplot(1, 3, 3)
plt.scatter(candidates[:, 0], candidates[:, 1], alpha=0.1, s=10, color='gray')
plt.scatter(best_iql[:, 0], best_iql[:, 1], color='red', s=30, label='IQL Selected')
circle = plt.Circle((0, 0), 1.0, color='black', alpha=0.3)
plt.gca().add_patch(circle)
plt.title(f"IQL Critic (Avg Dist: {torch.norm(best_iql, dim=1).mean():.2f})")
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.legend()
plt.grid(True)

plt.show()

# %% [markdown]
# ## 结论
#
# 观察上面的图：
# - **Raw Actor** 生成的点大部分在 **Distance ~ 1.8** 处（模仿了次优的演示数据）。
# - **IQL Selected (Red)** 生成的点应该显著地**更靠近障碍物边缘 (Distance ~ 1.2)**，即使演示数据中这样的样本非常少。
#
# 这证明了 System 2 (Flow Matching + IQL Critic) 具有**超越演示数据 (Better-than-Demonstrator)** 的能力。
