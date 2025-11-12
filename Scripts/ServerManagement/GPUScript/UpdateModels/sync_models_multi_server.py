#!/usr/bin/env python3
"""
Multi-Server Model Sync with NAS HTTP Server

Prerequisites:
- NAS HTTP server must be running on root@dt-thpc-nas01:8000
- Start server manually with: ./start_nas_http_server.sh

Workflow:
1. Check if NAS HTTP server is accessible
2. Load GPU servers from gpu_servers.txt
3. Download NAS sha256-list.csv as source of truth
4. For each GPU server:
   a. Update checksums on GPU server using compare_checksums.py
   b. Compare GPU server CSV with NAS CSV to get files to download
   c. Download missing/corrupted files from NAS HTTP server
   d. Update GPU server checksums again to verify
"""

import os
import sys
import subprocess
from pathlib import Path

# Configuration
NAS_HOST = "root@dt-thpc-nas01"
NAS_IP = "100.104.93.82"
NAS_PATH = "/zfs/data/official-models-ckpt-tensordata"
HTTP_PORT = 8000
SCRIPT_DIR = Path(__file__).parent
DEFAULT_GPU_PATH = "/mnt/models/official-models"  # Default path for GPU servers


def print_step(step, total, message):
    """Print step header"""
    print(f"\n{'='*70}")
    print(f"[Step {step}/{total}] {message}")
    print('='*70)


def load_gpu_servers(filepath="gpu_servers.txt"):
    """Load GPU servers from file

    Format: user@hostname
    Example: root@dfw-026-001

    The default path will be appended automatically.
    """
    servers = []

    if not os.path.exists(filepath):
        print(f"❌ Error: {filepath} not found")
        sys.exit(1)

    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue

            # Validate format: user@hostname
            if '@' not in line:
                print(f"⚠️  Line {line_num}: Invalid format (expected user@hostname) - {line}")
                continue

            # Append default path
            server_with_path = f"{line}:{DEFAULT_GPU_PATH}"
            servers.append(server_with_path)

    print(f"✅ Loaded {len(servers)} GPU server(s)")
    print(f"   Using default path: {DEFAULT_GPU_PATH}")
    return servers


def update_gpu_server_checksums(gpu_server):
    """Update checksums on GPU server using compare_checksums.py

    This will generate/update sha256-list.csv on the GPU server and
    clean up any corrupted files.

    Args:
        gpu_server: Server spec in format user@hostname:/path
    """
    print(f"\n📊 Updating checksums on {gpu_server}...")

    cmd = [
        'python3',
        str(SCRIPT_DIR / 'compare_checksums.py'),
        gpu_server
    ]

    # Stream output in real-time instead of capturing it
    result = subprocess.run(cmd, text=True)

    if result.returncode != 0:
        print(f"   ⚠️  Warning: Checksum update returned code {result.returncode}")

    # Extract hostname for CSV file
    hostname = gpu_server.split('@')[-1].split(':')[0]
    gpu_csv_local = f"sha256-list-{hostname}.csv"

    print(f"   ✅ GPU server checksums updated: {gpu_csv_local}")
    return gpu_csv_local


def get_files_to_download(gpu_csv_local, nas_csv_local):
    """Compare GPU server CSV with NAS CSV to get list of files to download

    Args:
        gpu_csv_local: Local path to GPU server CSV
        nas_csv_local: Local path to NAS CSV (source of truth)

    Returns:
        list: Filenames that need to be downloaded
    """
    print(f"\n📋 Comparing checksums to determine files to download...")
    print(f"   GPU server CSV: {gpu_csv_local}")
    print(f"   NAS CSV (source of truth): {nas_csv_local}")

    cmd = [
        'python3',
        str(SCRIPT_DIR / 'compare_checksums.py'),
        gpu_csv_local,
        nas_csv_local
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Parse output - each line is a filename to download
    files_to_download = []
    for line in result.stdout.strip().split('\n'):
        line = line.strip()
        if line and not line.startswith('#'):
            files_to_download.append(line)

    print(f"   ✅ Found {len(files_to_download)} file(s) to download")

    return files_to_download


def check_nas_http_server():
    """Check if NAS HTTP server is accessible

    Returns:
        bool: True if server is accessible, False otherwise
    """
    print(f"\n🌐 Checking NAS HTTP server...")
    print(f"   URL: http://{NAS_IP}:{HTTP_PORT}")

    # Try to download the CSV to verify server is accessible
    test_url = f"http://{NAS_IP}:{HTTP_PORT}/sha256-list.csv"

    result = subprocess.run(
        ['wget', '--spider', '-q', test_url],
        capture_output=True,
        timeout=10
    )

    if result.returncode == 0:
        print(f"   ✅ NAS HTTP server is accessible")
        return True
    else:
        print(f"   ❌ NAS HTTP server is NOT accessible")
        print(f"\n   Please start the NAS HTTP server first:")
        print(f"   On NAS: cd /root/utils && ./start_nas_http_server.sh")
        print(f"   Or run: ssh {NAS_HOST} 'cd /root/utils && ./start_nas_http_server.sh'")
        return False


def download_nas_csv():
    """Download sha256-list.csv from NAS HTTP server as source of truth

    Returns:
        str: Local path to downloaded CSV file
    """
    print(f"\n📥 Downloading NAS CSV (source of truth) via HTTP...")

    nas_csv_url = f"http://{NAS_IP}:{HTTP_PORT}/sha256-list.csv"
    nas_csv_local = SCRIPT_DIR / 'nas-sha256-list.csv'

    result = subprocess.run(
        ['wget', '-q', '-O', str(nas_csv_local), nas_csv_url],
        capture_output=True,
        text=True,
        timeout=30
    )

    if result.returncode != 0:
        print(f"   ❌ Failed to download NAS CSV")
        if result.stderr:
            print(f"   Error: {result.stderr}")
        return None

    print(f"   ✅ Downloaded to: {nas_csv_local}")
    return str(nas_csv_local)




def download_files_from_nas(gpu_server, files):
    """Download files from NAS HTTP server to GPU server one by one

    Uses wget -O to download and overwrite existing files automatically.

    Args:
        gpu_server: Server spec in format user@hostname:/path
        files: List of filenames to download

    Returns:
        int: Number of files successfully downloaded
    """
    if not files:
        print("   ✅ No files to download")
        return 0

    # Extract hostname and path
    hostname, path = gpu_server.split(':')
    http_url = f"http://{NAS_IP}:{HTTP_PORT}"

    print(f"\n📥 Downloading {len(files)} file(s) from NAS...")
    print(f"   From: {http_url}")
    print(f"   To: {gpu_server}")

    success_count = 0

    for i, filename in enumerate(files, 1):
        file_url = f"{http_url}/{filename}"
        dest_path = f"{path}/{filename}"

        # Show progress
        if i == 1 or i % 10 == 0 or i == len(files):
            print(f"   [{i}/{len(files)}] {filename}")

        # Download file using wget with progress bar
        # -O will overwrite existing file automatically (no need for rm)
        # --progress=bar:force shows progress bar even when redirected
        # --show-progress displays download speed and ETA
        wget_cmd = f'wget --progress=bar:force --show-progress "{file_url}" -O "{dest_path}"'

        result = subprocess.run(
            ['ssh', hostname, wget_cmd],
            text=True,
            timeout=3600  # 1 hour per file
        )

        if result.returncode == 0:
            success_count += 1
        else:
            print(f"   ⚠️  Failed: {filename}")
            if i <= 3:  # Show error for first few failures
                print(f"      Error: wget returned code {result.returncode}")

    print(f"   ✅ Downloaded {success_count}/{len(files)} file(s)")
    return success_count




def sync_single_server(gpu_server, nas_csv_local):
    """Sync a single GPU server with NAS

    Workflow:
    1. Update checksums on GPU server
    2. Compare checksums to get files to download
    3. Download files from NAS HTTP server
    4. Update checksums on GPU server again to verify

    Args:
        gpu_server: Server spec in format user@hostname:/path
        nas_csv_local: Local path to NAS CSV file (source of truth)

    Returns:
        bool: True if sync successful, False otherwise
    """
    print(f"\n{'='*70}")
    print(f"Syncing: {gpu_server}")
    print('='*70)

    try:
        # Step 1: Update checksums on GPU server
        print_step(1, 4, f"Update checksums on {gpu_server}")
        gpu_csv_local = update_gpu_server_checksums(gpu_server)

        # Step 2: Compare checksums to get files to download
        print_step(2, 4, "Compare checksums")
        files_to_download = get_files_to_download(gpu_csv_local, nas_csv_local)

        if not files_to_download:
            print("\n✅ Server already in sync - no files to download")
            return True

        # Step 3: Download files from NAS
        print_step(3, 4, "Download files from NAS")
        success_count = download_files_from_nas(gpu_server, files_to_download)

        if success_count < len(files_to_download):
            print(f"\n⚠️  Warning: Only {success_count}/{len(files_to_download)} files downloaded successfully")

        # Step 4: Update checksums on GPU server again to verify
        print_step(4, 4, f"Verify checksums on {gpu_server}")
        update_gpu_server_checksums(gpu_server)

        print(f"\n✅ Server sync completed: {success_count}/{len(files_to_download)} files downloaded")
        return True

    except Exception as e:
        print(f"\n❌ Error syncing {gpu_server}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*70)
    print("Multi-Server Model Sync with NAS HTTP Server")
    print("="*70)
    print(f"NAS Source: {NAS_HOST}:{NAS_PATH}")
    print(f"NAS HTTP: http://{NAS_IP}:{HTTP_PORT}")
    print(f"Server List: gpu_servers.txt")

    try:
        # Step 1: Check if NAS HTTP server is accessible
        print_step(1, 4, "Check NAS HTTP server")
        if not check_nas_http_server():
            print("\n❌ NAS HTTP server is not accessible. Exiting.")
            sys.exit(1)

        # Step 2: Load GPU servers
        print_step(2, 4, "Load GPU servers from gpu_servers.txt")
        servers = load_gpu_servers()

        if not servers:
            print("❌ No servers found in gpu_servers.txt")
            sys.exit(1)

        print(f"\nServers to sync:")
        for i, server in enumerate(servers, 1):
            print(f"  {i}. {server}")

        # Step 3: Download NAS CSV (source of truth) via HTTP
        print_step(3, 4, "Download NAS CSV (source of truth)")
        nas_csv_local = download_nas_csv()

        if not nas_csv_local:
            print("❌ Failed to download NAS CSV")
            sys.exit(1)

        # Step 4: Sync each GPU server
        print_step(4, 4, "Sync GPU servers")

        results = {}
        for i, server in enumerate(servers, 1):
            print(f"\n{'#'*70}")
            print(f"# Server {i}/{len(servers)}")
            print(f"{'#'*70}")

            success = sync_single_server(server, nas_csv_local)
            results[server] = success

        # Summary
        print("\n" + "="*70)
        print("Sync Summary")
        print("="*70)

        success_count = sum(1 for v in results.values() if v)

        for server, success in results.items():
            status = "✅ Success" if success else "❌ Failed"
            print(f"{status}: {server}")

        print(f"\nTotal: {success_count}/{len(servers)} servers synced successfully")
        print("="*70)

        if success_count == len(servers):
            print("\n🎉 All servers synced successfully!")
            sys.exit(0)
        else:
            print(f"\n⚠️  {len(servers) - success_count} server(s) failed to sync")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Sync interrupted by user")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
