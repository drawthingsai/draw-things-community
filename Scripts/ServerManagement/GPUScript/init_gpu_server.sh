#!/bin/bash

# Script to initialize GPU servers with Docker, CUDA, and dependencies
# Usage:
#   ./InitGPUServer.sh root@hostname
#   ./InitGPUServer.sh --dry-run root@hostname

# Exit immediately if a command exits with a non-zero status.
set -e

# Parse arguments
REMOTE_HOST=""
DRY_RUN=false

for arg in "$@"; do
    if [ "$arg" == "--dry-run" ] || [ "$arg" == "-n" ]; then
        DRY_RUN=true
    elif [ -z "$REMOTE_HOST" ]; then
        REMOTE_HOST="$arg"
    fi
done

# The initialization commands to run on the remote server
INIT_COMMANDS='
set -e

echo "=================================================="
echo "  GPU Server Initialization"
echo "=================================================="
echo ""

# --- 0. System & Docker Installation ---
echo "➡️ Step 1: Updating system and installing Docker..."

# Update package index and install prerequisites
sudo apt-get update
sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common wget

# Add Docker official GPG key
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add Docker repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Update package index again and install Docker
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io

echo "✅ Docker installed successfully."

# --- 1. Install CUDA and related dependencies ---

## Section 1: System Preparation
echo ""
echo "➡️ Step 2: Installing NVIDIA CUDA and Container Toolkit..."
sudo apt-get upgrade -y

## Section 2: Install CUDA Toolkit
echo "Adding NVIDIA CUDA repository..."

# Install the repository keyring
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
rm cuda-keyring_1.1-1_all.deb

echo "Installing CUDA Toolkit..."
sudo apt-get -y install cuda-toolkit

# Add CUDA to the system PATH for all users
echo "export PATH=/usr/local/cuda/bin\${PATH:+:\${PATH}}" | sudo tee /etc/profile.d/cuda.sh
echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}" | sudo tee -a /etc/profile.d/cuda.sh

echo "✅ CUDA Toolkit installed successfully."

## Section 3: Install NVIDIA Container Toolkit
echo "Adding NVIDIA Container Toolkit repository..."
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg --yes
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed "s#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g" | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update

echo "Installing NVIDIA Container Toolkit..."
sudo apt-get install -y nvidia-container-toolkit

## Section 4: Configure Docker
echo "Configuring Docker to use the NVIDIA runtime..."
sudo nvidia-ctk runtime configure --runtime=docker

# Restart the Docker daemon to apply the new configuration
sudo systemctl restart docker

echo "✅ NVIDIA Container Toolkit installed and configured."

# --- 3. Security & Directory Setup ---
echo ""
echo "➡️ Step 4: Applying security settings..."

# Lock the root user password for enhanced security
sudo passwd -l root
echo "✅ Root password locked."

# --- 4. Install Python dependencies ---
echo ""
echo "➡️ Step 5: Installing Python and dependencies..."

sudo apt-get update
sudo apt-get install -y python3-pip python3-venv nvtop htop

# Create a virtual environment for Python packages
python3 -m venv /opt/draw-things-venv
/opt/draw-things-venv/bin/pip install --upgrade pip
/opt/draw-things-venv/bin/pip install tqdm schedule boto3

# Add venv to PATH for all users
echo "export PATH=/opt/draw-things-venv/bin:\$PATH" | sudo tee /etc/profile.d/draw-things-venv.sh
sudo chmod +x /etc/profile.d/draw-things-venv.sh
source /etc/profile.d/draw-things-venv.sh

echo "✅ Python dependencies installed in /opt/draw-things-venv"

# --- 5. Network Performance Tuning ---
echo ""
echo "➡️ Step 6: Configuring network performance settings..."

# Apply TCP buffer and BBR settings immediately
sudo sysctl -w net.ipv4.tcp_rmem="4096 87380 134217728"
sudo sysctl -w net.ipv4.tcp_wmem="4096 65536 134217728"
sudo sysctl -w net.core.rmem_max=134217728
sudo sysctl -w net.core.wmem_max=134217728
sudo sysctl -w net.core.default_qdisc=fq
sudo sysctl -w net.ipv4.tcp_congestion_control=bbr

# Make settings persistent across reboots
cat <<EOF | sudo tee /etc/sysctl.d/99-network-performance.conf
# TCP buffer sizes for high-bandwidth connections
net.ipv4.tcp_rmem = 4096 87380 134217728
net.ipv4.tcp_wmem = 4096 65536 134217728
net.core.rmem_max = 134217728
net.core.wmem_max = 134217728

# BBR congestion control
net.core.default_qdisc = fq
net.ipv4.tcp_congestion_control = bbr
EOF

echo "✅ Network performance settings configured and persisted."

# --- 6. Pull Docker images ---
echo ""
echo "➡️ Step 7: Pulling Docker images..."

sudo docker pull drawthingsai/draw-things-grpc-server-cli:latest

echo "✅ Docker images pulled."

# --- 7. Final Summary ---
echo ""
echo "=================================================="
echo "🎉 GPU Server Initialization Complete!"
echo "=================================================="
echo ""
echo "A system **reboot** is highly recommended."
echo ""
echo "Post-reboot verification commands:"
echo "  1. nvidia-smi                    # Check GPU status"
echo "  2. nvcc --version                # Check CUDA version"
echo "  3. sudo docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi"
echo ""
echo "=================================================="
'

# Function to perform dry-run check on a single server
dry_run_server() {
    local HOST="$1"

    echo "Target: $HOST"
    echo ""

    # Check if ssh is available
    if ! command -v ssh &> /dev/null; then
        echo "Error: ssh command not found. Please install openssh-client."
        return 1
    fi

    # Test SSH connection
    echo "Testing SSH connection to $HOST..."
    if ! ssh -o BatchMode=yes -o ConnectTimeout=10 "$HOST" "echo 'Connection successful'" 2>/dev/null < /dev/null; then
        echo "Error: Cannot connect to $HOST"
        echo "Please ensure:"
        echo "  1. The host is reachable"
        echo "  2. SSH keys are set up for passwordless authentication"
        echo "  3. You have root access"
        return 1
    fi
    echo "✅ SSH connection successful"
    echo ""

    # Get system information
    echo "--- System Information ---"
    ssh "$HOST" "lsb_release -a 2>/dev/null || cat /etc/os-release 2>/dev/null || echo 'Could not determine OS'" < /dev/null
    echo ""

    echo "--- GPU Information ---"
    ssh "$HOST" "nvidia-smi 2>/dev/null || echo 'nvidia-smi not available (GPU driver may not be installed yet)'" < /dev/null
    echo ""

    echo "--- Commands that would be executed ---"
    echo "$INIT_COMMANDS" | head -80
    echo "... (truncated, see script for full commands)"
    echo ""

    return 0
}

# Function to initialize a single server
init_server() {
    local HOST="$1"

    echo "Target: $HOST"
    echo ""

    # Check if ssh is available
    if ! command -v ssh &> /dev/null; then
        echo "Error: ssh command not found. Please install openssh-client."
        return 1
    fi

    # Test SSH connection
    echo "Testing SSH connection to $HOST..."
    if ! ssh -o BatchMode=yes -o ConnectTimeout=10 "$HOST" "echo 'Connection successful'" 2>/dev/null < /dev/null; then
        echo "Error: Cannot connect to $HOST"
        echo "Please ensure:"
        echo "  1. The host is reachable"
        echo "  2. SSH keys are set up for passwordless authentication"
        echo "  3. You have root access"
        return 1
    fi
    echo "✅ SSH connection successful"
    echo ""

    # Execute initialization commands on remote server
    echo "Starting initialization on $HOST..."
    echo ""
    ssh -t "$HOST" "$INIT_COMMANDS" < /dev/null

    return 0
}

# Validate arguments
if [ -z "$REMOTE_HOST" ]; then
    echo "Usage:"
    echo "  $0 root@hostname"
    echo "  $0 --dry-run root@hostname"
    echo ""
    echo "Options:"
    echo "  --dry-run, -n   Check server connectivity and show system info without making changes"
    echo ""
    echo "This script will initialize a GPU server with:"
    echo "  - Docker"
    echo "  - CUDA Toolkit"
    echo "  - NVIDIA Container Toolkit"
    echo "  - Python dependencies (tqdm, schedule, boto3)"
    echo "  - Network performance tuning (TCP buffers, BBR)"
    echo "  - Draw Things gRPC server Docker image"
    exit 1
fi

if [ "$DRY_RUN" = true ]; then
    echo "=================================================="
    echo "  [DRY RUN] Check GPU Server"
    echo "=================================================="

    if dry_run_server "$REMOTE_HOST"; then
        echo ""
        echo "=================================================="
        echo "✅ Dry run check passed!"
        echo ""
        echo "Run without --dry-run to perform the actual initialization."
        exit 0
    else
        echo ""
        echo "=================================================="
        echo "❌ Dry run check failed!"
        exit 1
    fi
else
    echo "=================================================="
    echo "  Initialize GPU Server"
    echo "=================================================="

    if init_server "$REMOTE_HOST"; then
        echo ""
        echo "=================================================="
        echo "✅ Initialization complete!"
        echo ""
        echo "Remember to reboot the server."
        exit 0
    else
        echo ""
        echo "=================================================="
        echo "❌ Initialization failed!"
        exit 1
    fi
fi
