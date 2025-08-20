from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchinfo import summary

from ae_models import SymmetricAutoencoder
from ae_pca import compute_pca_95


def flatten_to_vector(t: torch.Tensor) -> torch.Tensor:
    return t.view(-1)


def get_datasets(data_root: str = "./data"):
    # Inputs normalized to [-1, 1] to match decoder Tanh output
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Lambda(flatten_to_vector),  # flatten 28x28 -> 784
    ])
    train = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    test = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)
    return train, test


def train_autoencoder(model: nn.Module, device: torch.device, train_loader: DataLoader,
                      optimizer: torch.optim.Optimizer, num_epochs: int) -> List[float]:
    per_batch_loss: List[float] = []
    criterion = nn.MSELoss()
    model.train()
    for epoch in range(num_epochs):
        epoch_avg = 0.0
        start = time.time()
        for inputs, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs = inputs.to(device)
            outputs, _ = model(inputs)
            loss = criterion(outputs, inputs)
            per_batch_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_avg += loss.item()
        epoch_avg /= max(1, len(train_loader))
        elapsed = time.time() - start
        print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss per Batch: {epoch_avg:.6f}, Time: {elapsed:.2f}s")
    return per_batch_loss


def evaluate_autoencoder(model: nn.Module, device: torch.device, data_loader: DataLoader, criterion: nn.Module) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            outputs, _ = model(inputs)
            loss = criterion(outputs, inputs)
            total_loss += loss.item()
    return total_loss / max(1, len(data_loader))


def plot_training_loss(per_batch_loss: List[float], out_path: Path):
    plt.figure(figsize=(6, 4))
    plt.plot(per_batch_loss)
    plt.xlabel("Batch")
    plt.ylabel("MSE Loss")
    plt.title("Autoencoder Training Loss")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)


def plot_reconstructions(model: nn.Module, device: torch.device, data_loader: DataLoader, out_path: Path, num_images: int = 10):
    model.eval()
    images, _ = next(iter(data_loader))
    images = images[:num_images]
    with torch.no_grad():
        outputs, _ = model(images.to(device))
    # invert normalization from [-1,1] back to [0,1]
    originals = (images * 0.5 + 0.5).cpu().view(-1, 28, 28)
    recons = (outputs.cpu() * 0.5 + 0.5).view(-1, 28, 28)

    plt.figure(figsize=(num_images * 1.2, 2.6))
    for i in range(num_images):
        plt.subplot(2, num_images, i + 1)
        plt.imshow(originals[i], cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.ylabel('Input')
        plt.subplot(2, num_images, num_images + i + 1)
        plt.imshow(recons[i], cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.ylabel('Recon')
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)


def main():
    parser = argparse.ArgumentParser(description="Train MNIST symmetric autoencoder")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--code-size", type=str, default="auto", help="int or 'auto' to use PCA 95%")
    parser.add_argument("--hidden-sizes", type=str, default="2500", help="comma-separated hidden sizes between input and code")
    parser.add_argument("--device", type=str, default="auto", help="auto|cuda|mps|cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds, test_ds = get_datasets(args.data_root)
    input_size = 28 * 28

    if args.code_size == "auto":
        n95, _ = compute_pca_95(max_samples=60000, data_root=args.data_root, seed=args.seed)
        code_size = n95
    else:
        code_size = int(args.code_size)

    hidden_sizes = [int(h) for h in args.hidden_sizes.split(",") if h.strip()]
    layer_sizes = [input_size] + hidden_sizes + [code_size]

    # Select device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    model = SymmetricAutoencoder(layer_sizes).to(device)
    # Ensure summary runs on the same device, and keep model on target device afterwards
    print(summary(model, input_size=(args.batch_size, input_size), device=str(device)))
    model = model.to(device)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    per_batch_loss = train_autoencoder(model, device, train_loader, optimizer, args.epochs)

    criterion = nn.MSELoss()
    val_loss = evaluate_autoencoder(model, device, test_loader, criterion)
    print(f"Test (reconstruction) MSE: {val_loss:.6f}")

    # Outputs
    torch.save(model.state_dict(), out_dir / "mnist_ae.pt")
    plot_training_loss(per_batch_loss, out_dir / "training_loss.png")
    plot_reconstructions(model, device, test_loader, out_dir / "reconstructions.png")


if __name__ == "__main__":
    main()


