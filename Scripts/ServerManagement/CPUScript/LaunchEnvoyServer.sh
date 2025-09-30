#!/bin/bash

# This script starts the Envoy proxy container.
# It requires the path to the envoy-config.yaml file as an argument.
# It should be run with sudo privileges.

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Argument Validation ---
# Check if exactly one argument is provided.
if [ "$#" -ne 1 ]; then
  echo "❌ Error: Missing Envoy configuration file path."
  echo ""
  echo "Usage:   $0 <PATH_TO_ENVOY_CONFIG>"
  echo "Example: $0 /home/user/envoy/envoy-config.yaml"
  exit 1
fi

# --- Configuration ---
ENVOY_CONFIG_PATH="$1"

# Check if the provided Envoy config file actually exists.
if [ ! -f "$ENVOY_CONFIG_PATH" ]; then
  echo "❌ Error: The Envoy configuration file was not found at the specified path:"
  echo "   $ENVOY_CONFIG_PATH"
  exit 1
fi

echo "✅ Envoy configuration file found."

# Start the Envoy proxy container
echo "Stopping Envoy proxy container..."
docker stop envoy_grpc_web_proxy || true
echo "Removing Envoy proxy container..."
docker rm envoy_grpc_web_proxy || true

echo "🚀 Starting Envoy proxy container..."
docker run -d \
  --name envoy_grpc_web_proxy \
  --network host \
  --restart unless-stopped \
  -v "$ENVOY_CONFIG_PATH:/etc/envoy/envoy.yaml:ro" \
  -v /etc/letsencrypt:/etc/letsencrypt:ro \
  envoyproxy/envoy:v1.28-latest

echo "✅ Envoy proxy service has been started."