from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


def compute_pca_95(max_samples: Optional[int] = None, data_root: str = "./data", batch_size: int = 1024,
                   num_workers: int = 0, seed: int = 0):
    rng = np.random.default_rng(seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t.view(-1)),  # flatten to 784
    ])
    train = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)

    indices = np.arange(len(train))
    if max_samples is not None and max_samples > 0:
        indices = rng.choice(indices, size=min(max_samples, len(train)), replace=False)

    subset = Subset(train, indices.tolist())
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    xs = []
    for xb, _ in loader:
        xs.append(xb.numpy())
    X = np.concatenate(xs, axis=0)

    pca = PCA()
    pca.fit(X)
    explained = np.cumsum(pca.explained_variance_ratio_)
    n_components_95 = int(np.searchsorted(explained, 0.95, side='left') + 1)
    return n_components_95, explained


## removed custom subset; using torch.utils.data.Subset for multiprocessing safety


def main():
    parser = argparse.ArgumentParser(description="Compute PCA and cumulative explained variance on MNIST")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--max-samples", type=int, default=60000)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n95, explained = compute_pca_95(args.max_samples, args.data_root, seed=args.seed, num_workers=args.workers)

    # Save results
    (out_dir / "n_components_95.txt").write_text(str(n95))
    np.save(out_dir / "explained_cumsum.npy", explained)

    # Plot
    plt.figure(figsize=(6, 4))
    ks = np.arange(1, len(explained) + 1)
    plt.plot(ks, explained, label="Cumulative explained variance")
    plt.axhline(0.95, color="red", linestyle="--", label="95%")
    plt.axvline(n95, color="green", linestyle="--", label=f"k={n95}")
    plt.xlabel("Number of components k")
    plt.ylabel("Cumulative explained variance")
    plt.title("MNIST PCA cumulative explained variance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "pca_cumulative_variance.png", dpi=160)

    print(f"Number of components to explain 95% variance: {n95}")


if __name__ == "__main__":
    main()


