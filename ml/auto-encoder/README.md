# MNIST 对称自编码器（Autoencoder）复现

本项目最小可运行地复现了 Matt Faulkner 博客中关于自编码器的实验流程：
- 使用 PCA 估计 95% 累积方差所需维度以指导瓶颈层大小
- 对称全连接自编码器（Encoder/Decoder 镜像）；除最后一层外均使用 ReLU；Decoder 末层使用 Tanh
- 输入按 [-1, 1] 归一化，与 Tanh 输出范围匹配
- 使用 Adam 训练，并输出训练损失曲线与重建可视化

参考来源：[Deep Learning: Autoencoders (2024-07-26)](https://mfaulk.github.io/2024/07/26/pytorch-ae.html)

---

## 环境与安装

- Python ≥ 3.9
- 建议使用虚拟环境（任选其一）：

venv：
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

conda：
```bash
conda create -n ae python=3.10 -y
conda activate ae
pip install -r requirements.txt
```

> 说明：`torchvision` 可能提示 `torchvision.io.image` 的 `libjpeg` 警告，本项目未使用该模块读取图像，可忽略该警告。

---

## 快速开始

1) 计算 PCA 累积方差与 95% 维度：
```bash
python ae_pca.py --output-dir outputs --max-samples 60000 --workers 0
```
产物：
- outputs/pca_cumulative_variance.png
- outputs/n_components_95.txt（MNIST 通常约为 154）

2) 训练自编码器（默认自动读取 PCA 的 95% 维度作为瓶颈大小）：
```bash
python ae_train.py \
  --output-dir outputs \
  --epochs 10 \
  --batch-size 256 \
  --hidden-sizes 2500 \
  --lr 1e-3 \
  --device auto \
  --workers 0
```
产物：
- outputs/mnist_ae.pt（模型权重）
- outputs/training_loss.png（训练损失曲线）
- outputs/reconstructions.png（重建效果）

固定瓶颈维度（如 150）以便与文中示例对照：
```bash
python ae_train.py --code-size 150 --hidden-sizes 2500 --epochs 10 --batch-size 256 --lr 1e-3 --device auto --workers 0
```

---

## 命令行参数

- --output-dir：输出目录，默认 outputs
- --data-root：数据目录（自动下载 MNIST），默认 ./data
- --batch-size：批大小，默认 256
- --epochs：训练轮数，默认 10
- --lr：学习率，默认 1e-3
- --workers：DataLoader 线程数，默认 0（macOS/MPS/多进程 pickling 更稳）
- --hidden-sizes：输入与瓶颈之间的隐藏层宽度（逗号分隔），默认 2500
- --code-size：瓶颈维度，整数或 auto（默认），auto 会调用 PCA 并使用 95% 方差对应维度
- --device：auto|cuda|mps|cpu，默认 auto
  - Apple Silicon 建议 mps 或 auto
  - NVIDIA 建议 cuda 或 auto

---

## 项目结构

- ae_models.py：对称自编码器 SymmetricAutoencoder
- ae_pca.py：计算 PCA 累积方差与 95% 维度，并绘图
- ae_train.py：训练与评估脚本，输出损失曲线与重建图
- requirements.txt：依赖列表

---

## 设计与复现要点

- 输入预处理：ToTensor 后做 Normalize((0.5,), (0.5,))，将像素缩放到 [-1, 1]
- 模型结构：Encoder 与 Decoder 镜像；除最后线性层外使用 ReLU；Decoder 末层使用 Tanh
- PCA 指导瓶颈：MNIST 上 95% 方差对应维度约为 154（见参考博文与本项目 ae_pca.py 实测）
- 训练：使用 Adam（MSE 重建损失），支持自动选择设备（CUDA/MPS/CPU）

---

## 常见问题（FAQ）

- 看到 torchvision.io.image 的 libjpeg 警告？
  - 不影响本项目运行，可忽略。

- macOS 上多进程 DataLoader 报 pickling 错误？
  - 将 --workers 设为 0（默认已为 0）。

- Apple Silicon（MPS）卡顿或报设备不匹配？
  - 使用 --device auto 或显式 --device mps。本项目在打印模型结构时已确保与训练设备一致。

---

## 参考

- 原文与思路来源：[Deep Learning: Autoencoders (2024-07-26)](https://mfaulk.github.io/2024/07/26/pytorch-ae.html)
