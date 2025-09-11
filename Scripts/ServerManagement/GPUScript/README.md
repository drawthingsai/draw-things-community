# GPU Script Setup Guide

## Initial Setup

After SSH into the server:

```bash
./InitGPUServer.sh  # Sets up Docker, CUDA, and Docker image
```

## Model Synchronization

### Option 1: Cloudflare R2 Sync (Recommended)
```bash
python3 /path/to/GPUScript/r2_sync_verification.py \
  -p /mnt/disk2/official-models/ \
  --account-id <your-account-id> \
  --access-key <your-access-key> \
  --secret-key <your-secret-key>
```

### Option 2: Rsync from Another Server
Use rsync with SSH pubkey authentication. Note: Must use physical IP address instead of Tailscale IP (10x slower).

## Testing Setup

1. Create test Docker container:
```bash
sudo docker run -it --name test3 \
  -v /path/to/official-models/:/models \
  -v /path/to/utils/:/utils \
  --gpus all \
  drawthingsai/draw-things-grpc-server-cli /bin/bash
```

2. Format tensor data (inside container):
```bash
./utils/TensorDataFormatter2 -p /models/ \
  --skip "qwen_image_edit_1.0_bf16.ckpt, qwen_image_1.0_bf16.ckpt"
```
**Note:** bf16 models may crash - issue pending investigation.

## Tailscale Setup script (Optional, you could setup tailscale directly)

```bash
/home/wlin1/Drawthings/draw-things/Scripts/ServerManagement/InitTailscale.sh
```

## Launch GPU Server

Ensure Tailscale is configured, then run:

```bash
/path/to/GPUScript/LaunchGPUServer.sh <tailscale ip address> <models_path> <utils_path> <lora_models_path>
```

### Example:
```bash
/home/utils/GPUScript/LaunchGPUServer.sh \
  100.168.1.100 \
  /disk1/official-models/ \
  /root/utils \
  /disk2/loraModels/
```