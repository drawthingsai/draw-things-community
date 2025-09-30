#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- 0. System & Docker Installation ---
echo "➡️ Step 1: Updating system and installing Docker..."

# Update package index and install prerequisites
apt update
apt install -y apt-transport-https ca-certificates curl software-properties-common

# Add Docker's official GPG key
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc

# Add Docker repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  tee /etc/apt/sources.list.d/docker.list > /dev/null

# Update package index again and install Docker
apt update
apt install -y docker-ce docker-ce-cli containerd.io

# Add current user to the docker group (requires logout/login to take effect)
if [ -n "$SUDO_USER" ]; then
    usermod -aG docker "$SUDO_USER"
    echo "✅ User $SUDO_USER added to the docker group. You'll need to log out and back in for this to take effect."
fi
echo "✅ Docker installed successfully."

# --- 1. Install CUDA and related dependencies ---

## Section 1: System Preparation
echo "🚀 Starting installation of NVIDIA CUDA and Container Toolkit..."
echo "Updating package lists and upgrading existing packages..."
sudo apt-get update
sudo apt-get upgrade -y

## Section 2: Install CUDA Toolkit
echo "✅ Adding NVIDIA CUDA repository..."

# Install the repository keyring
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
rm cuda-keyring_1.1-1_all.deb

echo "Installing CUDA Toolkit..."
# This command installs the latest stable CUDA toolkit
sudo apt-get -y install cuda-toolkit

# Add CUDA to the system's PATH for all users
echo 'export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}' | sudo tee /etc/profile.d/cuda.sh
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' | sudo tee -a /etc/profile.d/cuda.sh

echo "CUDA Toolkit installed successfully."

## Section 3: Install NVIDIA Container Toolkit
echo "✅ Adding NVIDIA Container Toolkit repository..."
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update

echo "Installing NVIDIA Container Toolkit..."
sudo apt-get install -y nvidia-container-toolkit

## Section 4: Configure Docker
echo "✅ Configuring Docker to use the NVIDIA runtime..."
# This command automatically detects and configures the container engine (Docker)
sudo nvidia-ctk runtime configure --runtime=docker

# Restart the Docker daemon to apply the new configuration
sudo systemctl restart docker

echo "NVIDIA Container Toolkit installed and configured."

## Section 5: Final Verification
echo ""
echo "🎉 Installation Complete!"
echo "--------------------------------------------------"
echo "A system **reboot** is highly recommended to ensure all changes take effect."
echo "" echo ""
echo "1. Check the NVIDIA driver and GPU status:"
echo "   nvidia-smi"
echo ""
echo "2. Check the CUDA compiler version:"
echo "   nvcc --version"
echo ""
echo "3. Test the Docker integration by running a CUDA container:"
echo "   sudo docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi"
echo "--------------------------------------------------"


# --- 2. Security & Directory Setup ---
echo "➡️ Step 2: Applying security settings and preparing directories..."

# Lock the root user's password for enhanced security
passwd -l root
echo "✅ Root password locked."

# --- 3. Start Services (Docker Containers) ---
echo "➡️ Step 3: Pulling grpc images and starting services..."

# Pull the required Docker images
docker pull drawthingsai/draw-things-grpc-server-cli:latest

# install pip and script dependencies 
sudo apt update
sudo apt install python3-pip
pip3 --version
pip install tqdm schedule boto3
sudo apt install nvtop htop
