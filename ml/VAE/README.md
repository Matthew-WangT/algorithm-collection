# 变分自编码器 (Variational Autoencoder, VAE)

本项目实现了变分自编码器，基于论文 ["Autoencoding Variational Bayes"](https://arxiv.org/abs/1312.6114) 和博客文章 [Deep Learning: Variational Autoencoders](https://mfaulk.github.io/2024/09/08/vae.html)。

## 概述

变分自编码器(VAE)是一种生成模型，它学习数据的潜在表示，并能够生成新的、逼真的样本。与标准自编码器不同，VAE使用概率编码器和解码器，并通过变分推理来近似难以处理的后验分布。

### 主要特点

- **概率编码器**: 将输入映射到潜在空间的均值和方差
- **重参数化技巧**: 使随机采样过程可微分
- **生成能力**: 可以从潜在空间采样生成新数据
- **正则化**: KL散度损失防止过拟合并确保潜在空间的平滑性

## 文件结构

```
VAE/
├── vae_models.py      # VAE模型实现
├── vae_train.py       # 训练脚本
├── requirements.txt   # 依赖包
└── README.md         # 本文档
```

## 模型架构

### 1. 全连接VAE (`VariationalAutoencoder`)
- 适用于扁平化的输入数据
- 编码器: 输入 → 隐藏层 → μ和log(σ²)
- 解码器: 潜在变量 → 隐藏层 → 重构输出

### 2. 卷积VAE (`ConvolutionalVAE`)
- 适用于图像数据
- 编码器: 卷积层 → 全连接层 → μ和log(σ²)
- 解码器: 潜在变量 → 转置卷积层 → 重构图像

## 损失函数

VAE的损失函数包含两部分：

1. **重构损失** (Reconstruction Loss): 衡量重构质量
   ```
   L_recon = BCE(x_recon, x)
   ```

2. **KL散度损失** (KL Divergence Loss): 正则化潜在空间
   ```
   L_KL = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
   ```

总损失：`L_total = L_recon + L_KL`

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 训练模型

```bash
# 使用卷积VAE训练 (推荐)
python vae_train.py --model-type conv --epochs 50 --code-size 20

# 使用全连接VAE训练
python vae_train.py --model-type fc --epochs 50 --code-size 20 --hidden-size 400

# 自定义参数
python vae_train.py \
    --model-type conv \
    --epochs 100 \
    --batch-size 128 \
    --lr 1e-3 \
    --code-size 10 \
    --output-dir my_results
```

### 参数说明

- `--model-type`: 模型类型，`fc` (全连接) 或 `conv` (卷积)
- `--epochs`: 训练轮数
- `--batch-size`: 批大小
- `--lr`: 学习率
- `--code-size`: 潜在空间维度
- `--hidden-size`: 隐藏层维度 (仅全连接模型)
- `--output-dir`: 输出目录
- `--device`: 设备选择 (`auto`, `cuda`, `mps`, `cpu`)

### 输出结果

训练完成后，会在输出目录生成以下文件：

- `vae_[model_type]_model.pt`: 训练好的模型
- `training_losses.png`: 训练损失曲线
- `reconstructions.png`: 重构结果对比
- `generated_samples.png`: 生成的新样本
- `latent_space.png`: 潜在空间可视化 (仅当code_size=2时)

## 理论背景

### 变分推理

VAE使用变分推理来近似难以处理的后验分布 p(z|x)。核心思想是：

1. 定义一个参数化的近似分布 q_φ(z|x)
2. 最小化 q_φ(z|x) 和 p(z|x) 之间的KL散度
3. 推导出Evidence Lower BOund (ELBO) 作为优化目标

### 重参数化技巧

为了使随机采样过程可微分，VAE使用重参数化技巧：

```
z = μ + σ * ε, 其中 ε ~ N(0,I)
```

这样梯度可以通过μ和σ反向传播，而不是通过随机采样操作。

### 生成过程

1. 从先验分布采样: z ~ p(z) = N(0,I)
2. 通过解码器生成: x ~ p_θ(x|z)

## 示例结果

### 重构质量
训练好的VAE能够很好地重构输入图像，保持主要特征的同时去除噪声。

### 生成样本
从潜在空间的标准正态分布采样，可以生成逼真的手写数字。

### 潜在空间
当潜在空间维度为2时，可以观察到不同数字类别在潜在空间中的分布。

## 扩展

### 1. β-VAE
在KL散度项前加权重β来控制潜在表示的解耦程度：
```python
total_loss = recon_loss + beta * kl_loss
```

### 2. 条件VAE (CVAE)
在编码器和解码器中加入条件信息（如类别标签）。

### 3. 其他数据集
修改数据加载部分可以应用到其他数据集，如CIFAR-10、CelebA等。

## 参考资料

1. [Autoencoding Variational Bayes](https://arxiv.org/abs/1312.6114) - 原始VAE论文
2. [Deep Learning: Variational Autoencoders](https://mfaulk.github.io/2024/09/08/vae.html) - 详细的VAE教程
3. [An Introduction to Variational Autoencoders](https://arxiv.org/abs/1906.02691) - VAE综述

## 许可证

本项目遵循MIT许可证。
