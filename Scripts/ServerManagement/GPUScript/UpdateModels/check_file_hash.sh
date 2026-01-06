#!/bin/bash
# Check SHA256 hash of a file across all GPU servers
# Usage: ./check_file_hash.sh <relative_file_path>
# Example: ./check_file_hash.sh sd15/v1-5-pruned-emaonly.safetensors

BASE_PATH="/mnt/models/official-models"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVERS_FILE="$SCRIPT_DIR/gpu_servers.txt"

if [ -z "$1" ]; then
    echo "Usage: $0 <relative_file_path>"
    echo "Example: $0 sd15/v1-5-pruned-emaonly.safetensors"
    exit 1
fi

FILE_PATH="$1"
FULL_PATH="$BASE_PATH/$FILE_PATH"

echo "Checking: $FULL_PATH"
echo "========================================"

# Run all servers in parallel
while read -r line; do
    server="${line%%|*}"
    hostname="${server#*@}"
    (echo "$hostname: $(ssh -o ConnectTimeout=10 "$server" "sha256sum \"$FULL_PATH\"" 2>&1)") &
done < <(grep -v '^#' "$SERVERS_FILE" | grep -v '^$')
wait
