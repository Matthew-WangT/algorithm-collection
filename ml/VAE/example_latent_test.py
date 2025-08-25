#!/usr/bin/env python3
"""
VAE Latent Code Test Example
Demonstrate how to manually specify latent code values and generate images
"""

import sys
import os
from pathlib import Path

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt
    import numpy as np
    from vae_models import VariationalAutoencoder, ConvolutionalVAE
except ImportError as e:
    print(f"Import failed: {e}")
    print("Please install dependencies: pip install -r requirements.txt")
    sys.exit(1)


def load_model_example():
    """Example of loading a model"""
    print("🔄 Loading Model Example")

    # Choose model file
    model_path = "outputs/vae_conv_model.pt"  # or "outputs/vae_fc_model.pt"

    if not os.path.exists(model_path):
        print(f"❌ Model file does not exist: {model_path}")
        print("Please run: python vae_train.py --model-type conv --epochs 10")
        return None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Get model information
    model_type = checkpoint['model_type']
    code_size = checkpoint['code_size']

    print(f"✅ Loaded model: {model_path}")
    print(f"   Type: {model_type}")
    print(f"   Latent code dimension: {code_size}")

    # Create model
    if model_type == "fc":
        model = VariationalAutoencoder(code_size=code_size)
    else:
        model = ConvolutionalVAE(code_size=code_size)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, {'model_type': model_type, 'code_size': code_size}


def single_latent_code_test():
    """Single latent code test"""
    print("\n🎯 Single Latent Code Test")

    model, model_info = load_model_example()
    if model is None:
        return

    device = next(model.parameters()).device
    code_size = model_info['code_size']

    # Example 1: All-zero latent code
    print("\n1️⃣ Testing all-zero latent code")
    latent_code = torch.zeros(1, code_size)
    print(f"Latent code: {latent_code.squeeze().tolist()}")

    with torch.no_grad():
        generated = model.decode(latent_code.to(device))
        plot_single_image(generated, "All-zero latent code")

    # Example 2: Random latent code
    print("\n2️⃣ Testing random latent code")
    latent_code = torch.randn(1, code_size)
    print(f"Latent code: {latent_code.squeeze().tolist()[:5]}...")  # Show first 5 values

    with torch.no_grad():
        generated = model.decode(latent_code.to(device))
        plot_single_image(generated, "Random latent code")

    # Example 3: Manually specified latent code
    print("\n3️⃣ Testing manually specified latent code")
    # Create an interesting latent code pattern
    values = []
    for i in range(code_size):
        if i % 3 == 0:
            values.append(1.0)    # Every 3rd dimension: 1
        elif i % 3 == 1:
            values.append(-0.5)   # Every 3rd+1 dimension: -0.5
        else:
            values.append(0.0)    # Every 3rd+2 dimension: 0

    latent_code = torch.tensor(values).unsqueeze(0)
    print(f"Latent code pattern: {values}")

    with torch.no_grad():
        generated = model.decode(latent_code.to(device))
        plot_single_image(generated, "Pattern latent code")


def multiple_latent_codes_test():
    """Multiple latent codes test"""
    print("\n🎯 Multiple Latent Codes Test")

    model, model_info = load_model_example()
    if model is None:
        return

    device = next(model.parameters()).device
    code_size = model_info['code_size']

    # 创建多个latent code
    latent_codes = []

    # 1. 全零
    latent_codes.append(torch.zeros(code_size))

    # 2. 第一个维度变化
    for i in [-2, -1, 0, 1, 2]:
        code = torch.zeros(code_size)
        code[0] = i
        latent_codes.append(code)

    # 3. 随机值
    latent_codes.append(torch.randn(code_size))

    # 4. 对称模式
    code = torch.zeros(code_size)
    for i in range(0, min(code_size, 10), 2):
        code[i] = 1.0
        if i + 1 < code_size:
            code[i + 1] = -1.0
    latent_codes.append(code)

    print(f"Testing {len(latent_codes)} latent codes")

    # Generate images
    latent_batch = torch.stack(latent_codes).to(device)

    with torch.no_grad():
        generated_batch = model.decode(latent_batch)

    # 绘制结果
    plot_multiple_images(generated_batch, latent_codes)


def plot_single_image(image_tensor, title):
    """Plot single image"""
    img = image_tensor.squeeze().cpu()

    # Handle different model output shapes
    if len(img.shape) == 1:  # Fully connected model
        img = img.view(28, 28)

    plt.figure(figsize=(4, 4))
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()


def plot_multiple_images(images_tensor, latent_codes):
    """Plot multiple images"""
    batch_size = images_tensor.shape[0]

    # Calculate grid layout
    cols = min(4, batch_size)
    rows = (batch_size + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

    if rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for i in range(batch_size):
        row, col = i // cols, i % cols

        img = images_tensor[i].squeeze().cpu()
        if len(img.shape) == 1:  # Fully connected model
            img = img.view(28, 28)

        axes[row, col].imshow(img, cmap='gray')
        axes[row, col].axis('off')

        # Create title
        latent_str = latent_codes[i].tolist()
        if len(latent_str) > 3:
            title = f"[{latent_str[0]:.1f}, {latent_str[1]:.1f}, {latent_str[2]:.1f}, ...]"
        else:
            title = f"{latent_str}"
        axes[row, col].set_title(title, fontsize=8)

    # Hide extra subplots
    for i in range(batch_size, rows * cols):
        row, col = i // cols, i % cols
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()


def latent_code_interpolation():
    """Latent code interpolation test"""
    print("\n🎯 Latent Code Interpolation Test")

    model, model_info = load_model_example()
    if model is None:
        return

    device = next(model.parameters()).device
    code_size = model_info['code_size']

    # Create two different latent codes
    code1 = torch.randn(code_size)
    code2 = torch.randn(code_size)

    print(f"Starting latent code: {code1.tolist()[:3]}...")
    print(f"Ending latent code: {code2.tolist()[:3]}...")

    # Create interpolation sequence
    num_steps = 8
    interpolated_codes = []

    for i in range(num_steps):
        alpha = i / (num_steps - 1)
        code = (1 - alpha) * code1 + alpha * code2
        interpolated_codes.append(code)

    # Generate images
    latent_batch = torch.stack(interpolated_codes).to(device)

    with torch.no_grad():
        generated_batch = model.decode(latent_batch)

    # Plot interpolation results
    fig, axes = plt.subplots(1, num_steps, figsize=(num_steps * 2, 2))

    for i in range(num_steps):
        img = generated_batch[i].squeeze().cpu()
        if len(img.shape) == 1:
            img = img.view(28, 28)

        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f"Step {i}", fontsize=8)

    plt.suptitle("Latent Code Interpolation", fontsize=12)
    plt.tight_layout()
    plt.show()


def main():
    """Main function"""
    print("🎨 VAE Latent Code Control Test")
    print("=" * 50)

    # Check if trained models exist
    conv_model = "outputs/vae_conv_model.pt"
    fc_model = "outputs/vae_fc_model.pt"

    if not (os.path.exists(conv_model) or os.path.exists(fc_model)):
        print("❌ No trained model files found")
        print("\nPlease train models first:")
        print("1. Train Convolutional VAE: python vae_train.py --model-type conv --epochs 10")
        print("2. Train Fully Connected VAE: python vae_train.py --model-type fc --epochs 10")
        return

    print("Available test functions:")
    print("1. Single latent code test")
    print("2. Multiple latent codes test")
    print("3. Latent code interpolation test")

    while True:
        print("\n" + "-" * 30)
        choice = input("Choose test function (1/2/3/q): ").strip()

        if choice == '1':
            single_latent_code_test()
        elif choice == '2':
            multiple_latent_codes_test()
        elif choice == '3':
            latent_code_interpolation()
        elif choice.lower() == 'q':
            break
        else:
            print("❌ Invalid choice, please enter 1, 2, 3 or q")


if __name__ == "__main__":
    main()
