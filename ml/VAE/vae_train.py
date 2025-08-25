from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchinfo import summary
import numpy as np

from vae_models import VariationalAutoencoder, ConvolutionalVAE, vae_loss


def get_datasets(data_root: str = "./data", model_type: str = "fc"):
    """
    获取MNIST数据集
    
    Args:
        data_root: 数据根目录
        model_type: 模型类型 ("fc" 全连接 或 "conv" 卷积)
        
    Returns:
        train_dataset, test_dataset
    """
    if model_type == "fc":
        # 全连接模型：展平并归一化到 [0, 1]
        transform = transforms.Compose([
            transforms.ToTensor(),  # [0, 1]
            transforms.Lambda(lambda x: x.view(-1))  # 展平 28x28 -> 784
        ])
    else:
        # 卷积模型：保持图像形状，归一化到 [0, 1]
        transform = transforms.Compose([
            transforms.ToTensor()  # [0, 1]
        ])
    
    train_dataset = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)
    
    return train_dataset, test_dataset


def train_vae(model: nn.Module, device: torch.device, train_loader: DataLoader,
              optimizer: optim.Optimizer, num_epochs: int) -> List[Tuple[float, float, float]]:
    """
    训练VAE模型
    
    Args:
        model: VAE模型
        device: 设备
        train_loader: 训练数据加载器
        optimizer: 优化器
        num_epochs: 训练轮数
        
    Returns:
        losses: 每个epoch的损失列表 [(total_loss, recon_loss, kl_loss), ...]
    """
    model.train()
    losses = []
    
    for epoch in range(num_epochs):
        epoch_total_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        
        start_time = time.time()
        
        for batch_idx, (data, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            data = data.to(device)
            
            # 前向传播
            x_recon, mu, log_var = model(data)
            
            # 计算损失
            total_loss, recon_loss, kl_loss = vae_loss(x_recon, data, mu, log_var)
            
            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # 累计损失
            epoch_total_loss += total_loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()
        
        # 计算平均损失
        num_samples = len(train_loader.dataset)
        avg_total_loss = epoch_total_loss / num_samples
        avg_recon_loss = epoch_recon_loss / num_samples
        avg_kl_loss = epoch_kl_loss / num_samples
        
        losses.append((avg_total_loss, avg_recon_loss, avg_kl_loss))
        
        elapsed_time = time.time() - start_time
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Total Loss: {avg_total_loss:.4f}")
        print(f"  Recon Loss: {avg_recon_loss:.4f}")
        print(f"  KL Loss: {avg_kl_loss:.4f}")
        print(f"  Time: {elapsed_time:.2f}s")
        print("-" * 50)
    
    return losses


def evaluate_vae(model: nn.Module, device: torch.device, test_loader: DataLoader) -> Tuple[float, float, float]:
    """
    评估VAE模型
    
    Args:
        model: VAE模型
        device: 设备
        test_loader: 测试数据加载器
        
    Returns:
        平均总损失, 平均重构损失, 平均KL损失
    """
    model.eval()
    total_loss = 0.0
    recon_loss = 0.0
    kl_loss = 0.0
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            x_recon, mu, log_var = model(data)
            
            loss_total, loss_recon, loss_kl = vae_loss(x_recon, data, mu, log_var)
            
            total_loss += loss_total.item()
            recon_loss += loss_recon.item()
            kl_loss += loss_kl.item()
    
    num_samples = len(test_loader.dataset)
    return total_loss / num_samples, recon_loss / num_samples, kl_loss / num_samples


def plot_training_losses(losses: List[Tuple[float, float, float]], out_path: Path):
    """
    绘制训练损失曲线
    
    Args:
        losses: 损失列表
        out_path: 输出路径
    """
    epochs = range(1, len(losses) + 1)
    total_losses, recon_losses, kl_losses = zip(*losses)
    
    plt.figure(figsize=(12, 4))
    
    # 总损失
    plt.subplot(1, 3, 1)
    plt.plot(epochs, total_losses, 'b-', label='Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Total Loss')
    plt.grid(True)
    
    # 重构损失
    plt.subplot(1, 3, 2)
    plt.plot(epochs, recon_losses, 'r-', label='Reconstruction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Reconstruction Loss')
    plt.grid(True)
    
    # KL散度损失
    plt.subplot(1, 3, 3)
    plt.plot(epochs, kl_losses, 'g-', label='KL Divergence Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('KL Divergence Loss')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_reconstructions(model: nn.Module, device: torch.device, test_loader: DataLoader, 
                        out_path: Path, num_images: int = 10, model_type: str = "fc"):
    """
    绘制重构结果
    
    Args:
        model: VAE模型
        device: 设备
        test_loader: 测试数据加载器
        out_path: 输出路径
        num_images: 显示图像数量
        model_type: 模型类型
    """
    model.eval()
    
    # 获取测试图像
    data, _ = next(iter(test_loader))
    data = data[:num_images].to(device)
    
    with torch.no_grad():
        x_recon, mu, log_var = model(data)
    
    # 转换为可视化格式
    if model_type == "fc":
        # 全连接模型：重塑为28x28
        originals = data.cpu().view(-1, 28, 28)
        reconstructions = x_recon.cpu().view(-1, 28, 28)
    else:
        # 卷积模型：去掉通道维度
        originals = data.cpu().squeeze(1)
        reconstructions = x_recon.cpu().squeeze(1)
    
    # 绘制对比图
    plt.figure(figsize=(num_images * 1.5, 3))
    
    for i in range(num_images):
        # 原图
        plt.subplot(2, num_images, i + 1)
        plt.imshow(originals[i], cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.ylabel('Original', fontsize=12)
        
        # 重构图
        plt.subplot(2, num_images, num_images + i + 1)
        plt.imshow(reconstructions[i], cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.ylabel('Reconstructed', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_generated_samples(model: nn.Module, device: torch.device, out_path: Path, 
                          num_samples: int = 64):
    """
    绘制生成的样本
    
    Args:
        model: VAE模型
        device: 设备
        out_path: 输出路径
        num_samples: 生成样本数量
    """
    model.eval()
    
    # 生成样本
    generated_samples = model.generate(num_samples, device)
    
    # 转换为可视化格式
    if len(generated_samples.shape) == 2:
        # 全连接模型输出
        samples = generated_samples.cpu().view(-1, 28, 28)
    else:
        # 卷积模型输出
        samples = generated_samples.cpu().squeeze(1)
    
    # 绘制网格
    grid_size = int(np.sqrt(num_samples))
    plt.figure(figsize=(10, 10))
    
    for i in range(min(num_samples, grid_size * grid_size)):
        plt.subplot(grid_size, grid_size, i + 1)
        plt.imshow(samples[i], cmap='gray')
        plt.axis('off')
    
    plt.suptitle('Generated Samples', fontsize=16)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_latent_space(model: nn.Module, device: torch.device, test_loader: DataLoader,
                     out_path: Path, model_type: str = "fc"):
    """
    可视化潜在空间 (仅当code_size=2时)
    
    Args:
        model: VAE模型
        device: 设备
        test_loader: 测试数据加载器
        out_path: 输出路径
        model_type: 模型类型
    """
    if model.code_size != 2:
        print(f"跳过潜在空间可视化：code_size={model.code_size} (需要code_size=2)")
        return
    
    model.eval()
    
    # 收集潜在表示和标签
    z_points = []
    labels = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            mu, log_var = model.encode(data)
            z_points.append(mu.cpu().numpy())
            labels.append(target.numpy())
    
    z_points = np.concatenate(z_points, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    # 绘制散点图
    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for digit in range(10):
        mask = labels == digit
        plt.scatter(z_points[mask, 0], z_points[mask, 1], 
                   c=[colors[digit]], label=f'Digit {digit}', alpha=0.6, s=20)
    
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title('Latent Space Visualization (Test Set)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="训练MNIST变分自编码器")
    parser.add_argument("--output-dir", type=str, default="outputs", help="输出目录")
    parser.add_argument("--data-root", type=str, default="./data", help="数据根目录")
    parser.add_argument("--batch-size", type=int, default=128, help="批大小")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--code-size", type=int, default=20, help="潜在空间维度")
    parser.add_argument("--hidden-size", type=int, default=400, help="隐藏层维度 (仅全连接模型)")
    parser.add_argument("--model-type", type=str, default="conv", choices=["fc", "conv"], 
                       help="模型类型：fc (全连接) 或 conv (卷积)")
    parser.add_argument("--device", type=str, default="auto", help="设备：auto|cuda|mps|cpu")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 创建输出目录
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 选择设备
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    
    print(f"使用设备: {device}")
    print(f"模型类型: {args.model_type}")
    print(f"潜在空间维度: {args.code_size}")
    
    # 获取数据集
    train_dataset, test_dataset = get_datasets(args.data_root, args.model_type)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 创建模型
    if args.model_type == "fc":
        model = VariationalAutoencoder(
            input_size=784,
            hidden_size=args.hidden_size,
            code_size=args.code_size
        )
    else:
        model = ConvolutionalVAE(code_size=args.code_size)
    
    model = model.to(device)
    
    # 显示模型信息
    if args.model_type == "fc":
        sample_input = torch.randn(args.batch_size, 784).to(device)
    else:
        sample_input = torch.randn(args.batch_size, 1, 28, 28).to(device)
    
    print("\n模型结构:")
    print(summary(model, input_data=sample_input, verbose=0))
    
    # 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 训练模型
    print("\n开始训练...")
    losses = train_vae(model, device, train_loader, optimizer, args.epochs)
    
    # 评估模型
    print("\n评估模型...")
    test_total_loss, test_recon_loss, test_kl_loss = evaluate_vae(model, device, test_loader)
    print(f"测试集 - 总损失: {test_total_loss:.4f}, 重构损失: {test_recon_loss:.4f}, KL损失: {test_kl_loss:.4f}")
    
    # 保存模型
    model_path = out_dir / f"vae_{args.model_type}_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': args.model_type,
        'code_size': args.code_size,
        'hidden_size': args.hidden_size if args.model_type == "fc" else None,
        'test_loss': test_total_loss
    }, model_path)
    print(f"模型已保存至: {model_path}")
    
    # 生成可视化图表
    print("\n生成可视化图表...")
    plot_training_losses(losses, out_dir / "training_losses.png")
    plot_reconstructions(model, device, test_loader, out_dir / "reconstructions.png", 
                        model_type=args.model_type)
    plot_generated_samples(model, device, out_dir / "generated_samples.png")
    plot_latent_space(model, device, test_loader, out_dir / "latent_space.png", 
                     model_type=args.model_type)
    
    print(f"\n训练完成！结果已保存至: {out_dir}")


if __name__ == "__main__":
    main()
