#!/bin/bash

# Script to start 8 separate Docker containers, each running one gRPCServerCLI process
# Each container will automatically restart if the process crashes
# Usage: ./script.sh <address> <models_path> <utils_path> <lora_models_path>

# Check if all required arguments are provided
if [ $# -ne 4 ]; then
    echo "Usage: $0 <address> <models_path> <utils_path> <lora_models_path>"
    echo "Example: $0 192.168.1.100 /disk1/official-models/ /root/utils /disk2/loraModels/"
    exit 1
fi

# Configuration from command line arguments
ADDRESS=$1
MODELS_PATH=$2
UTILS_PATH=$3
LORA_MODELS_PATH=$4

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

echo "Configuration:"
echo "  Address: $ADDRESS"
echo "  Models path: $MODELS_PATH"
echo "  Utils path: $UTILS_PATH"
echo "  LoRA models path: $LORA_MODELS_PATH"
echo ""

copy_tmpfs() {
    echo "Mount tmpfs..."
	$(UTILS_PATH)/mount_overlay.sh -s $MODELS_PATH -l $MODELS_PATH -f $(UTILS_PATH)/tmpfs.ls -m /mnt/unified-models/ -z 448G
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
        -v "${MODELS_PATH}:/models" \
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
