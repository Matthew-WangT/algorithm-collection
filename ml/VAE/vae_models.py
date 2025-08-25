from typing import Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class VariationalAutoencoder(nn.Module):
    """
    变分自编码器 (Variational Autoencoder, VAE)
    
    基于 "Autoencoding Variational Bayes" 论文实现
    使用重参数化技巧 (reparametrization trick) 来实现可微分的采样
    """
    
    def __init__(self, input_size: int = 784, hidden_size: int = 400, code_size: int = 20):
        """
        初始化VAE模型
        
        Args:
            input_size: 输入维度 (例如 MNIST 的 28*28=784)
            hidden_size: 隐藏层维度
            code_size: 潜在空间维度
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.code_size = code_size
        
        # 编码器 - 将输入映射到潜在空间的均值和方差
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # 潜在空间的均值和对数方差
        self.fc_mu = nn.Linear(hidden_size, code_size)
        self.fc_log_var = nn.Linear(hidden_size, code_size)
        
        # 解码器 - 将潜在变量重构为输入
        self.decoder = nn.Sequential(
            nn.Linear(code_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()  # 输出范围 [0, 1]
        )
    
    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        编码输入数据到潜在空间的均值和对数方差
        
        Args:
            x: 输入张量 (batch_size, input_size)
            
        Returns:
            mu_z: 潜在变量的均值
            log_var_z: 潜在变量的对数方差
        """
        h = self.encoder(x)
        mu_z = self.fc_mu(h)
        log_var_z = self.fc_log_var(h)
        return mu_z, log_var_z
    
    def reparametrize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """
        重参数化技巧：z = μ + σ * ε, 其中 ε ~ N(0,I)
        这允许梯度通过随机采样过程反向传播
        
        Args:
            mu: 均值
            log_var: 对数方差
            
        Returns:
            z: 采样的潜在变量
        """
        std = torch.exp(0.5 * log_var)  # σ = exp(log(σ²)/2)
        eps = torch.randn_like(std)     # ε ~ N(0,I)
        z = mu + eps * std              # 重参数化
        return z
    
    def decode(self, z: Tensor) -> Tensor:
        """
        解码潜在变量到重构数据
        
        Args:
            z: 潜在变量 (batch_size, code_size)
            
        Returns:
            x_recon: 重构的数据
        """
        return self.decoder(z)
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        VAE前向传播
        
        Args:
            x: 输入数据
            
        Returns:
            x_recon: 重构数据
            mu: 潜在变量均值
            log_var: 潜在变量对数方差
        """
        mu, log_var = self.encode(x)
        z = self.reparametrize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var
    
    def generate(self, num_samples: int, device: torch.device) -> Tensor:
        """
        从先验分布采样生成新数据
        
        Args:
            num_samples: 生成样本数量
            device: 设备
            
        Returns:
            generated_samples: 生成的样本
        """
        # 从标准正态分布采样
        z = torch.randn(num_samples, self.code_size).to(device)
        with torch.no_grad():
            generated_samples = self.decode(z)
        return generated_samples


class ConvolutionalVAE(nn.Module):
    """
    卷积变分自编码器
    适用于图像数据，基于博客中的实现
    """
    
    def __init__(self, code_size: int = 20):
        """
        初始化卷积VAE
        
        Args:
            code_size: 潜在空间维度
        """
        super().__init__()
        
        self.code_size = code_size
        
        # 编码器 - 卷积层
        self.encoder = nn.Sequential(
            # [B, 1, 28, 28] -> [B, 16, 14, 14]
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(True),
            
            # [B, 16, 14, 14] -> [B, 32, 7, 7]
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(True),
            
            # [B, 32, 7, 7] -> [B, 32*7*7]
            nn.Flatten(),
            
            # [B, 32*7*7] -> [B, 2*code_size] (均值和对数方差)
            nn.Linear(32 * 7 * 7, 2 * code_size)
        )
        
        # 解码器 - 转置卷积层
        self.decoder = nn.Sequential(
            # [B, code_size, 1, 1] -> [B, 32, 7, 7]
            nn.ConvTranspose2d(code_size, 32, 7),
            nn.ReLU(True),
            
            # [B, 32, 7, 7] -> [B, 16, 14, 14]
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            
            # [B, 16, 14, 14] -> [B, 1, 28, 28]
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # 输出范围 [0, 1]
        )
    
    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        编码输入图像到潜在空间的均值和对数方差
        
        Args:
            x: 输入图像 (batch_size, 1, 28, 28)
            
        Returns:
            mu_z: 潜在变量的均值
            log_var_z: 潜在变量的对数方差
        """
        mu_z, log_var_z = self.encoder(x).chunk(2, dim=1)
        return mu_z, log_var_z
    
    def reparametrize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """
        重参数化技巧
        
        Args:
            mu: 均值
            log_var: 对数方差
            
        Returns:
            z: 采样的潜在变量
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z: Tensor) -> Tensor:
        """
        解码潜在变量到重构图像
        
        Args:
            z: 潜在变量 (batch_size, code_size)
            
        Returns:
            x_recon: 重构的图像
        """
        # 重塑为 (batch_size, code_size, 1, 1) 用于转置卷积
        z = z.view(-1, self.code_size, 1, 1)
        return self.decoder(z)
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        卷积VAE前向传播
        
        Args:
            x: 输入图像
            
        Returns:
            x_recon: 重构图像
            mu: 潜在变量均值
            log_var: 潜在变量对数方差
        """
        mu, log_var = self.encode(x)
        z = self.reparametrize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var
    
    def generate(self, num_samples: int, device: torch.device) -> Tensor:
        """
        生成新图像
        
        Args:
            num_samples: 生成样本数量
            device: 设备
            
        Returns:
            generated_samples: 生成的图像
        """
        # 从标准正态分布采样
        z = torch.randn(num_samples, self.code_size).to(device)
        z = z.view(-1, self.code_size, 1, 1)
        with torch.no_grad():
            generated_samples = self.decoder(z)
        return generated_samples


def vae_loss(x_recon: Tensor, x: Tensor, mu: Tensor, log_var: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """
    VAE损失函数：重构损失 + KL散度损失
    
    基于ELBO (Evidence Lower BOund):
    log p(x) >= E[log p(x|z)] - KL(q(z|x) || p(z))
    
    Args:
        x_recon: 重构数据
        x: 原始数据
        mu: 潜在变量均值
        log_var: 潜在变量对数方差
        
    Returns:
        total_loss: 总损失
        recon_loss: 重构损失
        kl_loss: KL散度损失
    """
    # 重构损失 - 二元交叉熵
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
    
    # KL散度损失
    # 当先验 p(z) = N(0,I) 和近似后验 q(z|x) = N(μ,σ²I) 时的解析解
    # KL(q(z|x) || p(z)) = -0.5 * sum(1 + log(σ²) - μ² - σ²)
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    total_loss = recon_loss + kl_loss
    
    return total_loss, recon_loss, kl_loss
