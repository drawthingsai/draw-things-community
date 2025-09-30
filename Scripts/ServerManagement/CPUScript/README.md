# CPU Script Setup Guide

## Initial Setup

Initialize CPU server with dependencies (Docker, Docker image, Certbot, and Envoy):

```bash
/path/to/CPUScript/InitCPUServer.sh
```

This script installs all required dependencies and configures `envoy-config.yaml` for the gRPC web proxy.

## Certificate Setup

Initialize SSL certificates using Let's Encrypt:

```bash
/path/to/CPUScript/InitCertificate.sh
```

**Important:** Certificates are created in `/etc/letsencrypt/` and are required for both Envoy and the Proxy Server to function properly.

**Note:** Creating new certificates should not invalidate existing ones (per Gemini and Claude documentation), though this hasn't been fully tested.

## Tailscale Setup

Configure Tailscale for GPU server connectivity:

```bash
/path/to/InitTailscale.sh
```

**Required:** Tailscale must be configured for the CPU server to connect to GPU servers.

## Launch Services

### CPU Proxy Server

Start the ProxyServiceCLI CPU proxy server:

```bash
/path/to/CPUScript/LaunchCPUServer.sh <DATADOG_API_KEY> <PATH_TO_MODEL_LIST_FILE>
```

**Example:**
```bash
./LaunchCPUServer.sh 'dd_api_123abc...' /home/user/Documents/model-list
```

**Note:** This will stop and remove any existing container before starting a new one.

### Envoy Proxy Server

Start the Envoy gRPC web proxy:

```bash
/path/to/CPUScript/LaunchEnvoyServer.sh <PATH_TO_ENVOY_CONFIG>
```

**Example:**
```bash
./LaunchEnvoyServer.sh /home/user/envoy/envoy-config.yaml
```

**Note:** This will stop and remove any existing container before starting a new one.

## Model List

A default model list based on September 11, 2025 GPU server configurations is included for both services.