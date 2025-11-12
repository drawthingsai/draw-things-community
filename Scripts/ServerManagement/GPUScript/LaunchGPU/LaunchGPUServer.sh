#!/bin/bash

# Script to start 8 separate Docker containers, each running one gRPCServerCLI process
# Each container will automatically restart if the process crashes
# Usage:
#   Local:  ./LaunchGPUServer.sh <address> <models_path> <utils_path> <lora_models_path> [--skip-tmpfs]
#   Remote: ./LaunchGPUServer.sh <remote_host> <address> <models_path> <utils_path> <lora_models_path> [--skip-tmpfs]

# Exit immediately if a command exits with a non-zero status
set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"

# Check if all required arguments are provided
if [ $# -lt 4 ]; then
    echo "Error: Insufficient arguments"
    echo ""
    echo "Usage:"
    echo "  Local:  $SCRIPT_NAME <address> <models_path> <utils_path> <lora_models_path> [--skip-tmpfs]"
    echo "  Remote: $SCRIPT_NAME <remote_host> <address> <models_path> <utils_path> <lora_models_path> [--skip-tmpfs]"
    echo ""
    echo "Options:"
    echo "  --skip-tmpfs    Skip tmpfs overlay setup and use models path directly"
    echo ""
    echo "Examples:"
    echo "  Local:  $SCRIPT_NAME 192.168.1.100 /fast/models/official-models /root/utils /disk2/loraModels/"
    echo "  Remote: $SCRIPT_NAME root@hostname 192.168.1.100 /fast/models/official-models /root/utils /disk2/loraModels/"
    echo "  Skip tmpfs: $SCRIPT_NAME 192.168.1.100 /fast/models/official-models /root/utils /disk2/loraModels/ --skip-tmpfs"
    exit 1
fi

# Initialize skip tmpfs flag
SKIP_TMPFS=false

# Check if first argument is a remote host
REMOTE_HOST=""
if [[ "$1" =~ ^[a-zA-Z0-9_-]+@[a-zA-Z0-9._-]+$ ]]; then
    REMOTE_HOST="$1"
    shift
fi

# If we're executing remotely
if [ -n "$REMOTE_HOST" ]; then
    echo "=================================================="
    echo "  Remote GPU Server Launch"
    echo "=================================================="
    echo "Target: $REMOTE_HOST"
    echo ""

    # Check if ssh is available
    if ! command -v ssh &> /dev/null; then
        echo "Error: ssh command not found. Please install openssh-client."
        exit 1
    fi

    # Test SSH connection
    echo "Testing SSH connection to $REMOTE_HOST..."
    if ! ssh -o BatchMode=yes -o ConnectTimeout=5 "$REMOTE_HOST" "echo 'Connection successful'" 2>/dev/null; then
        echo "Error: Cannot connect to $REMOTE_HOST"
        echo "Please ensure:"
        echo "  1. The host is reachable"
        echo "  2. SSH keys are set up for passwordless authentication"
        echo "  3. You have appropriate access"
        exit 1
    fi
    echo "✅ SSH connection successful"
    echo ""

    # Parse remaining arguments for remote execution
    ADDRESS=$1
    MODELS_PATH=$2
    UTILS_PATH=$3
    LORA_MODELS_PATH=$4

    # Check for --skip-tmpfs flag
    if [ "$5" = "--skip-tmpfs" ]; then
        SKIP_TMPFS=true
    fi

    # Validate argument count
    if [ -z "$LORA_MODELS_PATH" ]; then
        echo "Error: Missing required arguments"
        echo "Usage: $SCRIPT_NAME $REMOTE_HOST <address> <models_path> <utils_path> <lora_models_path> [--skip-tmpfs]"
        exit 1
    fi

    echo "Configuration:"
    echo "  Address: $ADDRESS"
    echo "  Models path: $MODELS_PATH"
    echo "  Utils path: $UTILS_PATH"
    echo "  LoRA models path: $LORA_MODELS_PATH"
    echo "  Skip tmpfs: $SKIP_TMPFS"
    echo ""

    # Copy script and dependencies to remote server
    echo "Copying scripts to remote server..."

    # Create remote directory
    REMOTE_DIR="${UTILS_PATH}/LaunchGPU"
    ssh "$REMOTE_HOST" "mkdir -p $REMOTE_DIR"

    # Copy all files from the script directory
    scp -q "${SCRIPT_DIR}/"* "$REMOTE_HOST:$REMOTE_DIR/"
    ssh "$REMOTE_HOST" "chmod +x $REMOTE_DIR/*.sh"

    echo "✅ Scripts copied to $REMOTE_HOST:$REMOTE_DIR"
    echo ""

    # Build remote command
    REMOTE_CMD="cd $REMOTE_DIR && sudo bash $SCRIPT_NAME $ADDRESS $MODELS_PATH $UTILS_PATH $LORA_MODELS_PATH"
    if [ "$SKIP_TMPFS" = true ]; then
        REMOTE_CMD="$REMOTE_CMD --skip-tmpfs"
    fi

    # Execute the script remotely
    echo "Executing GPU server launch on $REMOTE_HOST..."
    echo "=================================================="
    ssh -t "$REMOTE_HOST" "$REMOTE_CMD"

    echo ""
    echo "=================================================="
    echo "✅ Remote GPU server launch complete!"
    echo "=================================================="
    exit 0
fi

# If we reach here, we're running locally
# Configuration from command line arguments
ADDRESS=$1
MODELS_PATH=$2
UTILS_PATH=$3
LORA_MODELS_PATH=$4

# Check for --skip-tmpfs flag
if [ "$5" = "--skip-tmpfs" ]; then
    SKIP_TMPFS=true
fi

# Validate that paths exist
if [ ! -d "$MODELS_PATH" ]; then
    echo "Error: Models path '$MODELS_PATH' does not exist"
    exit 1
fi

if [ ! -d "$UTILS_PATH" ]; then
    echo "Error: Utils path '$UTILS_PATH' does not exist"
    exit 1
fi

if [ ! -d "$LORA_MODELS_PATH" ]; then
    echo "Error: LoRA models path '$LORA_MODELS_PATH' does not exist"
    exit 1
fi

echo "=================================================="
echo "  Local GPU Server Launch"
echo "=================================================="
echo "Configuration:"
echo "  Address: $ADDRESS"
echo "  Models path: $MODELS_PATH"
echo "  Utils path: $UTILS_PATH"
echo "  LoRA models path: $LORA_MODELS_PATH"
echo "  Skip tmpfs: $SKIP_TMPFS"
echo ""

OVERLAY_MOUNT="/mnt/unified-models"
TMPFS_FILE_LIST="${SCRIPT_DIR}/tmpfs.ls"

# Function to setup overlay using mount_overlay.sh
setup_overlay() {
    echo "Setting up overlay filesystem..."

    # Check if mount_overlay.sh exists in the same directory as this script
    local mount_script="${SCRIPT_DIR}/mount_overlay.sh"
    if [ ! -f "$mount_script" ]; then
        echo "  ⚠️ mount_overlay.sh not found at $mount_script"
        echo "  Skipping overlay setup, will use $MODELS_PATH directly"
        return 1
    fi

    # Check if tmpfs.ls file exists
    if [ ! -f "$TMPFS_FILE_LIST" ]; then
        echo "  ⚠️ File list not found at $TMPFS_FILE_LIST"
        echo "  Skipping overlay setup, will use $MODELS_PATH directly"
        return 1
    fi

    echo "  Running: sudo $mount_script -s $MODELS_PATH -l $MODELS_PATH -f $TMPFS_FILE_LIST -m $OVERLAY_MOUNT -z 448G"

    sudo "$mount_script" \
        -s "$MODELS_PATH" \
        -l "$MODELS_PATH" \
        -f "$TMPFS_FILE_LIST" \
        -m "$OVERLAY_MOUNT" \
        -z 448G

    if [ $? -eq 0 ]; then
        echo "  ✅ Overlay mounted successfully at $OVERLAY_MOUNT"
        return 0
    else
        echo "  ❌ Failed to mount overlay"
        return 1
    fi
}

# Function to check if all GPUs are idle using nvidia-smi
check_gpu_idle() {
    # Check if nvidia-smi is available
    if ! command -v nvidia-smi &> /dev/null; then
        echo "Warning: nvidia-smi not found, skipping GPU usage check"
        return 0
    fi
    
    # Get GPU usage from nvidia-smi (simple and reliable)
    local gpu_usage=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
    
    if [ -z "$gpu_usage" ]; then
        echo "Warning: Could not read GPU usage from nvidia-smi, proceeding anyway"
        return 0
    fi
    
    # Check if all GPUs show 0% usage
    local all_idle=true
    local gpu_count=0
    local usage_list=""
    
    while IFS= read -r usage; do
        if [ -n "$usage" ]; then
            gpu_count=$((gpu_count + 1))
            usage_list="$usage_list$usage% "
            if [ "$usage" -gt 0 ]; then
                all_idle=false
            fi
        fi
    done <<< "$gpu_usage"
    
    if [ "$gpu_count" -eq 0 ]; then
        echo "Warning: No GPU usage data found, proceeding anyway"
        return 0
    fi
    
    if [ "$all_idle" = true ]; then
        echo "All $gpu_count GPUs are idle (0% usage)"
        return 0
    else
        echo "Some GPUs are still busy (usage: $usage_list)"
        return 1
    fi
}

# Wait for all GPUs to be idle (with 5-minute timeout)
wait_for_gpu_idle() {
    echo "Checking GPU usage before starting containers..."
    local timeout=300  # 5 minutes in seconds
    local elapsed=0
    local check_interval=10  # Check every 10 seconds
    
    while [ $elapsed -lt $timeout ]; do
        if check_gpu_idle; then
            echo "All GPUs are idle, proceeding with container startup"
            return 0
        fi
        
        echo "Waiting for GPUs to become idle... (${elapsed}s/${timeout}s elapsed)"
        sleep $check_interval
        elapsed=$((elapsed + check_interval))
    done
    
    echo "Timeout reached (5 minutes), proceeding with container startup anyway"
    return 0
}

# Pull the latest image
echo "Pulling latest Docker image..."
sudo docker pull drawthingsai/draw-things-grpc-server-cli:latest

# Wait for GPUs to be idle before starting containers
wait_for_gpu_idle

# Stop and remove any existing containers
echo "Stopping and removing existing containers..."
for i in $(seq 0 7); do
    CONTAINER_NAME="grpc_service_$i"
    sudo docker stop $CONTAINER_NAME 2>/dev/null || true
    sudo docker rm $CONTAINER_NAME 2>/dev/null || true
done

# Setup overlay filesystem
echo ""
echo "=================================================="
echo "  Overlay Filesystem Setup"
echo "=================================================="

if [ "$SKIP_TMPFS" = true ]; then
    echo "⏭️  Skipping tmpfs overlay setup (--skip-tmpfs flag)"
    MODELS_MOUNT_PATH="$MODELS_PATH"
    echo "Containers will use direct path: $MODELS_MOUNT_PATH"
else
    if setup_overlay; then
        MODELS_MOUNT_PATH="$OVERLAY_MOUNT"
        echo "✅ Containers will use overlay at: $MODELS_MOUNT_PATH"
    else
        MODELS_MOUNT_PATH="$MODELS_PATH"
        echo "⚠️  Containers will use direct path: $MODELS_MOUNT_PATH"
    fi
fi

echo "=================================================="
echo ""

# Start 4 separate containers
for i in $(seq 0 7); do
    PORT=$((40001 + i))
    GPU=$i
    CONTAINER_NAME="grpc_service_$i"
    
    echo "Starting container $CONTAINER_NAME on port $PORT using GPU $GPU at address $ADDRESS"
    
    sudo docker run -d \
        --name $CONTAINER_NAME \
        --network=host \
        --restart=unless-stopped \
        --gpus '"device='$GPU'"' \
        --tmpfs /tmp \
        -v "${MODELS_MOUNT_PATH}:/models" \
        -v "${UTILS_PATH}:/utils" \
        -v "${LORA_MODELS_PATH}:/loraModels" \
        drawthingsai/draw-things-grpc-server-cli:latest \
        /utils/start_single_grpc.sh $PORT $GPU $ADDRESS
        
    if [ $? -eq 0 ]; then
        echo "Successfully started container $CONTAINER_NAME"
    else
        echo "Failed to start container $CONTAINER_NAME"
    fi
    
    # Small delay between container starts
    sleep 2
done

echo "All containers started. Use 'sudo docker ps' to check status."
echo "To view logs for a specific container, use: sudo docker logs grpc_service_<number>"
echo "To stop all containers, run: sudo docker stop \$(sudo docker ps -q --filter name=grpc_service_)"
