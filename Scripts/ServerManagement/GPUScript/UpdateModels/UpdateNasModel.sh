#!/bin/bash

# NAS Model Update Script
# Usage:
#   Local:  ./UpdateNasModel.sh -p /path/to/models -u /path/to/utils -m MODE --account-id ID --access-key KEY --secret-key SECRET
#   Remote: ./UpdateNasModel.sh root@hostname -p /path/to/models -u /path/to/utils -m MODE --account-id ID --access-key KEY --secret-key SECRET
#
# Required parameters:
#   -p, --path          Path to models directory (e.g., /zfs/data/official-models-ckpt-tensordata/)
#   -u, --utils         Path to utils directory (e.g., /root/utils/)
#   -m, --mode          Operation mode: sync, format, checksum, or all (default: all)
#   --account-id        Cloudflare R2 account ID (required for sync mode)
#   --access-key        R2 access key (required for sync mode)
#   --secret-key        R2 secret key (required for sync mode)

# Exit immediately if a command exits with a non-zero status
set -e

# Initialize variables
REMOTE_HOST=""
MODELS_PATH=""
UTILS_PATH=""
MODE="all"
ACCOUNT_ID=""
ACCESS_KEY=""
SECRET_KEY=""

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"

# Parse arguments
if [[ $# -lt 1 ]]; then
    echo "Error: Insufficient arguments"
    echo ""
    echo "Usage:"
    echo "  Local:  $SCRIPT_NAME -p /path/to/models -u /path/to/utils -m MODE --account-id ID --access-key KEY --secret-key SECRET"
    echo "  Remote: $SCRIPT_NAME root@hostname -p /path/to/models -u /path/to/utils -m MODE --account-id ID --access-key KEY --secret-key SECRET"
    echo ""
    echo "Run '$SCRIPT_NAME --help' for more information"
    exit 1
fi

# Show help
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    cat << 'EOF'
NAS Model Update Script

Usage:
  Local:  ./UpdateNasModel.sh -p /path/to/models -u /path/to/utils -m MODE --account-id ID --access-key KEY --secret-key SECRET
  Remote: ./UpdateNasModel.sh root@hostname -p /path/to/models -u /path/to/utils -m MODE --account-id ID --access-key KEY --secret-key SECRET

This script performs operations on NAS model storage:
  1. sync     - Syncs and verifies models from Cloudflare R2 storage
  2. format   - Formats models using TensorDataFormatter
  3. checksum - Compares checksums to verify integrity
  4. all      - Runs all three operations in sequence (default)

Required parameters:
  -p, --path          Path to models directory (e.g., /zfs/data/official-models-ckpt-tensordata/)
  -u, --utils         Path to utils directory (e.g., /root/utils/)
  -m, --mode          Operation mode: sync, format, checksum, or all (default: all)
  --account-id        Cloudflare R2 account ID (required for sync mode)
  --access-key        R2 access key (required for sync mode)
  --secret-key        R2 secret key (required for sync mode)

Examples:
  # Run all operations locally
  ./UpdateNasModel.sh -p /zfs/data/official-models-ckpt-tensordata/ -u /root/utils/ \
    --account-id YOUR_ID --access-key YOUR_KEY --secret-key YOUR_SECRET

  # Run all operations on remote server
  ./UpdateNasModel.sh root@nas-server -p /zfs/data/official-models-ckpt-tensordata/ -u /root/utils/ \
    --account-id YOUR_ID --access-key YOUR_KEY --secret-key YOUR_SECRET

  # Only sync from R2
  ./UpdateNasModel.sh -p /data/models/ -u /root/utils/ -m sync \
    --account-id YOUR_ID --access-key YOUR_KEY --secret-key YOUR_SECRET

  # Only format (no R2 credentials needed)
  ./UpdateNasModel.sh -p /data/models/ -u /root/utils/ -m format

  # Only verify checksums (no R2 credentials needed)
  ./UpdateNasModel.sh -p /data/models/ -u /root/utils/ -m checksum

EOF
    exit 0
fi

# Check if first argument is a remote host
if [[ "$1" =~ ^[a-zA-Z0-9_-]+@[a-zA-Z0-9._-]+$ ]]; then
    REMOTE_HOST="$1"
    shift
fi

# Parse remaining arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--path)
            MODELS_PATH="$2"
            shift 2
            ;;
        -u|--utils)
            UTILS_PATH="$2"
            shift 2
            ;;
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        --account-id)
            ACCOUNT_ID="$2"
            shift 2
            ;;
        --access-key)
            ACCESS_KEY="$2"
            shift 2
            ;;
        --secret-key)
            SECRET_KEY="$2"
            shift 2
            ;;
        *)
            echo "Error: Unknown parameter: $1"
            echo "Run '$SCRIPT_NAME --help' for usage information"
            exit 1
            ;;
    esac
done

# Validate required parameters
if [[ -z "$MODELS_PATH" ]]; then
    echo "Error: Models path is required (-p or --path)"
    exit 1
fi

if [[ -z "$UTILS_PATH" ]]; then
    echo "Error: Utils path is required (-u or --utils)"
    exit 1
fi

# Validate mode
if [[ "$MODE" != "all" && "$MODE" != "sync" && "$MODE" != "format" && "$MODE" != "checksum" ]]; then
    echo "Error: Invalid mode '$MODE'. Must be one of: all, sync, format, checksum"
    exit 1
fi

# Validate R2 credentials if sync mode is enabled
if [[ "$MODE" == "all" || "$MODE" == "sync" ]]; then
    if [[ -z "$ACCOUNT_ID" || -z "$ACCESS_KEY" || -z "$SECRET_KEY" ]]; then
        echo "Error: R2 credentials are required for mode '$MODE'"
        echo "Please provide --account-id, --access-key, and --secret-key"
        exit 1
    fi
fi

# Function to run commands locally
run_local() {
    echo "=================================================="
    echo "  Local NAS Model Update"
    echo "=================================================="
    echo "Models path: $MODELS_PATH"
    echo "Utils path:  $UTILS_PATH"
    echo "Mode:        $MODE"
    echo ""

    # Validate paths exist
    if [[ ! -d "$MODELS_PATH" ]]; then
        echo "Error: Models path does not exist: $MODELS_PATH"
        exit 1
    fi

    if [[ ! -d "$UTILS_PATH" ]]; then
        echo "Error: Utils path does not exist: $UTILS_PATH"
        exit 1
    fi

    # Step 1: R2 Sync Verification
    if [[ "$MODE" == "all" || "$MODE" == "sync" ]]; then
        echo "➡️  Step 1: Running R2 sync verification..."
        echo "=================================================="

        R2_SCRIPT="$UTILS_PATH/r2_sync_verification.py"
        if [[ ! -f "$R2_SCRIPT" ]]; then
            echo "Error: R2 sync script not found: $R2_SCRIPT"
            exit 1
        fi

        python3 "$R2_SCRIPT" \
            -p "$MODELS_PATH" \
            --account-id "$ACCOUNT_ID" \
            --access-key "$ACCESS_KEY" \
            --secret-key "$SECRET_KEY"

        echo "✅ R2 sync verification completed"
        echo ""
    fi

    # Step 2: TensorData Formatting
    if [[ "$MODE" == "all" || "$MODE" == "format" ]]; then
        echo "➡️  Step 2: Running TensorData formatter..."
        echo "=================================================="

        FORMATTER="$UTILS_PATH/TensorDataFormatter"
        if [[ ! -f "$FORMATTER" ]]; then
            echo "Error: TensorDataFormatter not found: $FORMATTER"
            exit 1
        fi

        "$FORMATTER" --path "$MODELS_PATH"

        echo "✅ TensorData formatting completed"
        echo ""
    fi

    # Step 3: Checksum Comparison
    if [[ "$MODE" == "all" || "$MODE" == "checksum" ]]; then
        echo "➡️  Step 3: Comparing checksums..."
        echo "=================================================="

        CHECKSUM_SCRIPT="$UTILS_PATH/compare_checksums.py"
        if [[ ! -f "$CHECKSUM_SCRIPT" ]]; then
            echo "Error: Checksum comparison script not found: $CHECKSUM_SCRIPT"
            exit 1
        fi

        python3 "$CHECKSUM_SCRIPT" "$MODELS_PATH"

        echo "✅ Checksum comparison completed"
        echo ""
    fi

    echo "=================================================="
    echo "✅ NAS Model Update Complete!"
    echo "=================================================="
}

# Function to run commands remotely
run_remote() {
    echo "=================================================="
    echo "  Remote NAS Model Update"
    echo "=================================================="
    echo "Target:      $REMOTE_HOST"
    echo "Models path: $MODELS_PATH"
    echo "Utils path:  $UTILS_PATH"
    echo "Mode:        $MODE"
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

    # Build the remote command
    REMOTE_CMD="bash $UTILS_PATH/$SCRIPT_NAME -p \"$MODELS_PATH\" -u \"$UTILS_PATH\" -m \"$MODE\""

    # Add R2 credentials if provided
    if [[ -n "$ACCOUNT_ID" ]]; then
        REMOTE_CMD="$REMOTE_CMD --account-id \"$ACCOUNT_ID\""
    fi

    if [[ -n "$ACCESS_KEY" ]]; then
        REMOTE_CMD="$REMOTE_CMD --access-key \"$ACCESS_KEY\""
    fi

    if [[ -n "$SECRET_KEY" ]]; then
        REMOTE_CMD="$REMOTE_CMD --secret-key \"$SECRET_KEY\""
    fi

    # Copy this script to remote if it doesn't exist
    echo "Ensuring script is available on remote server..."
    ssh "$REMOTE_HOST" "mkdir -p $UTILS_PATH"
    scp -q "$0" "$REMOTE_HOST:$UTILS_PATH/$SCRIPT_NAME"
    ssh "$REMOTE_HOST" "chmod +x $UTILS_PATH/$SCRIPT_NAME"
    echo "✅ Script copied to remote"
    echo ""

    # Execute the script remotely
    echo "Executing update on $REMOTE_HOST..."
    echo "=================================================="
    ssh -t "$REMOTE_HOST" "$REMOTE_CMD"

    echo ""
    echo "=================================================="
    echo "✅ Remote update complete!"
    echo "=================================================="
}

# Main execution
if [[ -n "$REMOTE_HOST" ]]; then
    run_remote
else
    run_local
fi
