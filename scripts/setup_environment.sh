#!/bin/bash
set -e

echo "================================================="
echo "   OMNIDRIVE AI - Environment Setup Script"
echo "================================================="

# 1. Check OS (Ubuntu 22.04 required)
OS_VERSION=$(grep -oP '(?<=^VERSION_ID=")[^"]+' /etc/os-release || echo "")
if [ "$OS_VERSION" != "22.04" ]; then
    echo "[WARNING] This script is optimized for Ubuntu 22.04. Current OS version: $OS_VERSION"
else
    echo "[OK] Ubuntu 22.04 detected."
fi

# 2. Check CUDA installation
if ! command -v nvcc &> /dev/null; then
    echo "[WARNING] CUDA (nvcc) not found in PATH."
else
    echo "[OK] CUDA detected: $(nvcc --version | grep release | awk '{print $5}')"
fi

# 3. Check Docker + NVIDIA container toolkit
if ! command -v docker &> /dev/null; then
    echo "[ERROR] Docker is not installed. Please install Docker first."
    exit 1
else
    echo "[OK] Docker is installed."
fi

if ! dpkg -l | grep -q nvidia-container-toolkit; then
    echo "[WARNING] nvidia-container-toolkit not found. GPU passthrough to Docker might fail."
else
    echo "[OK] NVIDIA container toolkit detected."
fi

# 4. Install CAN utilities
echo "[INFO] Installing can-utils..."
sudo apt-get update
sudo apt-get install -y can-utils

# 5. Set up virtual CAN for testing
echo "[INFO] Setting up virtual CAN (vcan0)..."
sudo modprobe vcan
if ! ip link show vcan0 &> /dev/null; then
    sudo ip link add dev vcan0 type vcan
    sudo ip link set up vcan0
    echo "[OK] vcan0 created."
else
    echo "[OK] vcan0 already exists."
fi

# 6. Initialize git submodules
echo "[INFO] Initializing git submodules (autoware, drive_jepa, cardreamer, ijepa, vjepa, alpamayo_recipes)..."
git submodule update --init --recursive

# 7. Create Python virtual environment
echo "[INFO] Creating Python virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# 8. Install Python package in editable mode
echo "[INFO] Installing OMNIDRIVE Python package in editable mode..."
pip install --upgrade pip
if [ -f pyproject.toml ]; then
    pip install -e .
else
    echo "[WARNING] pyproject.toml not found. Skipping pip install -e ."
fi

# 9. Create data directory structure
echo "[INFO] Creating data directories..."
mkdir -p data/nuscenes data/waymo logs/inference logs/autoware checkpoints configs/autoware

# 10. Download pretrained weights
echo "[INFO] Downloading V-JEPA pretrained weights..."
mkdir -p weights/vjepa
# Placeholder for actual HF download link
wget -nc -O weights/vjepa/vjepa_weights.pth "https://huggingface.co/facebook/v-jepa/resolve/main/vjepa_weights.pth" || echo "[WARNING] Failed to download V-JEPA weights. Please download manually."

# 11. Set up wandb
echo "[INFO] Setting up Weights & Biases (wandb)..."
pip install wandb
echo "Please enter your wandb API key (or press Enter to skip):"
read -s WANDB_API_KEY
if [ -n "$WANDB_API_KEY" ]; then
    wandb login "$WANDB_API_KEY"
    echo "[OK] wandb logged in."
else
    echo "[INFO] Skipped wandb login."
fi

# 12. Setup Summary
echo "================================================="
echo "Setup Complete!"
echo "Data directories created at: ./data, ./logs, ./checkpoints, ./weights"
echo "Virtual environment ready at: .venv"
echo "Activate it using: source .venv/bin/activate"
echo "================================================="
