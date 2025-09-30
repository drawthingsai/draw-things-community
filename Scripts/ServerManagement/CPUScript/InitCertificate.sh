#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
# The domain is now hardcoded.
DOMAIN="compute.drawthings.ai"

# Prompt user for the remaining required information
read -p "Enter your email address (for Let's Encrypt notifications): " EMAIL

# --- 2. Certbot (Let's Encrypt) Installation ---
echo "➡️ Step 2: Installing Certbot to obtain SSL certificate..."

# Install snapd if not present
if ! command -v snap &> /dev/null; then
    apt install -y snapd
fi

# Install Certbot via snap
snap install certbot --classic

echo "✅ Certbot installed. Requesting certificate for $DOMAIN..."
# Obtain the certificate using standalone mode (requires port 80 to be free)
certbot certonly --standalone --non-interactive --agree-tos --email "$EMAIL" -d "$DOMAIN"
echo "✅ SSL certificate obtained successfully."