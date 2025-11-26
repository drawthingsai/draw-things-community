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

Usage:
  # Perform actual sync (sequential)
  python3 sync_models_multi_server.py

  # Parallel sync (all servers at once)
  python3 sync_models_multi_server.py --parallel

  # Dry-run mode (show what would be done without making changes)
  python3 sync_models_multi_server.py --dry-run

  # Parallel dry-run
  python3 sync_models_multi_server.py --parallel --dry-run

Notes:
  - In parallel mode, detailed output goes to sync-{hostname}.log files
  - Terminal shows real-time progress updates every 30 seconds
  - wget uses dot format (--progress=dot:mega) for cleaner logs
"""

import os
import sys
import subprocess
import argparse
import threading
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
NAS_HOST = "root@dt-thpc-nas01"
NAS_IP = "64.71.166.2"
NAS_PATH = "/zfs/data/official-models-ckpt-tensordata"
HTTP_PORT = 61767
SCRIPT_DIR = Path(__file__).parent
DEFAULT_GPU_PATH = "/mnt/models/official-models"  # Default path for GPU servers

# Global dry-run flag
DRY_RUN = False

# Global progress tracking for parallel mode
PROGRESS_LOCK = threading.Lock()
PROGRESS_DATA = {}  # {server: {"phase": str, "files_synced": int, "total_files": int, "status": str}}


def print_step(step, total, message):
    """Print step header"""
    print(f"\n{'='*70}")
    print(f"[Step {step}/{total}] {message}")
    print('='*70)


def update_progress(server, phase, files_synced=0, total_files=0, status="running"):
    """Update progress data for a server (thread-safe)

    Args:
        server: Server identifier
        phase: Current phase (e.g., "Checking checksums", "Downloading", "Verifying")
        files_synced: Number of files synced so far
        total_files: Total number of files to sync
        status: Status ("running", "completed", "failed")
    """
    with PROGRESS_LOCK:
        PROGRESS_DATA[server] = {
            "phase": phase,
            "files_synced": files_synced,
            "total_files": total_files,
            "status": status
        }


def log_print(log_file, message, also_print=False):
    """Write message to log file and optionally to stdout

    Args:
        log_file: File object to write to
        message: Message to write
        also_print: If True, also print to stdout
    """
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()
    if also_print or not log_file:
        print(message)


def display_progress_status():
    """Display current progress status for all servers (thread-safe)"""
    with PROGRESS_LOCK:
        if not PROGRESS_DATA:
            return

        print("\n" + "="*70)
        print("Current Status:")
        print("="*70)

        for server, data in sorted(PROGRESS_DATA.items()):
            phase = data["phase"]
            files_synced = data["files_synced"]
            total_files = data["total_files"]
            status = data["status"]

            # Format status line
            if status == "completed":
                status_icon = "✅"
                if total_files > 0:
                    status_text = f"{status_icon} Completed ({files_synced}/{total_files} files synced)"
                else:
                    status_text = f"{status_icon} Completed (already in sync)"
            elif status == "failed":
                status_icon = "❌"
                status_text = f"{status_icon} Failed"
            else:
                if total_files > 0:
                    status_text = f"{phase}... ({files_synced}/{total_files} files)"
                else:
                    status_text = f"{phase}..."

            # Get log filename
            hostname = server.split('@')[-1].split(':')[0]
            log_file_name = f"sync-{hostname}.log"

            print(f"[{hostname:15}] {status_text:40} | Log: {log_file_name}")

        print("="*70)


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


def update_gpu_server_checksums(gpu_server, log_file=None):
    """Update checksums on GPU server using compare_checksums.py

    This will generate/update sha256-list.csv on the GPU server and
    clean up any corrupted files.

    In dry-run mode, it only downloads the existing CSV without modifying it.

    Args:
        gpu_server: Server spec in format user@hostname:/path
        log_file: Optional file object to write logs to
    """
    log_print(log_file, f"\n📊 Updating checksums on {gpu_server}...")

    # Extract hostname for CSV file
    hostname = gpu_server.split('@')[-1].split(':')[0]
    gpu_csv_local = f"sha256-list-{hostname}.csv"

    if DRY_RUN:
        log_print(log_file, f"   [DRY RUN] Downloading existing checksums (read-only)")
        # In dry-run, just download the existing CSV from the server without modifying it
        # Extract user@host and path from gpu_server (format: user@host:/path)
        user_host = gpu_server.split(':')[0]  # e.g., root@dt-thpc-001
        server_path = gpu_server.split(':')[1]  # e.g., /mnt/models/official-models
        remote_csv = f"{user_host}:{server_path}/sha256-list.csv"

        result = subprocess.run(
            ['scp', remote_csv, gpu_csv_local],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            log_print(log_file, f"   ✅ Downloaded existing checksums to: {gpu_csv_local}")
        else:
            log_print(log_file, f"   ⚠️  Could not download checksums (file may not exist on server)")
            log_print(log_file, f"   [DRY RUN] In actual run, checksums would be generated on server")

        return gpu_csv_local

    cmd = [
        'python3',
        str(SCRIPT_DIR / 'compare_checksums.py'),
        gpu_server
    ]

    # Stream output to log file if provided, otherwise to stdout
    if log_file:
        result = subprocess.run(cmd, text=True, stdout=log_file, stderr=log_file)
    else:
        result = subprocess.run(cmd, text=True)

    if result.returncode != 0:
        log_print(log_file, f"   ⚠️  Warning: Checksum update returned code {result.returncode}")

    log_print(log_file, f"   ✅ GPU server checksums updated: {gpu_csv_local}")
    return gpu_csv_local


def get_files_to_download(gpu_csv_local, nas_csv_local, log_file=None):
    """Compare GPU server CSV with NAS CSV to get list of files to download

    Args:
        gpu_csv_local: Local path to GPU server CSV
        nas_csv_local: Local path to NAS CSV (source of truth)
        log_file: Optional file object to write logs to

    Returns:
        list: Filenames that need to be downloaded
    """
    log_print(log_file, f"\n📋 Comparing checksums to determine files to download...")
    log_print(log_file, f"   GPU server CSV: {gpu_csv_local}")
    log_print(log_file, f"   NAS CSV (source of truth): {nas_csv_local}")

    # Check if GPU CSV exists locally
    if not Path(gpu_csv_local).exists():
        log_print(log_file, f"   ⚠️  GPU server CSV not found: {gpu_csv_local}")
        if DRY_RUN:
            log_print(log_file, f"   [DRY RUN] Cannot compare - CSV not available")
        return []

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

    if DRY_RUN:
        log_print(log_file, f"   ✅ Found {len(files_to_download)} file(s) that would be downloaded")
    else:
        log_print(log_file, f"   ✅ Found {len(files_to_download)} file(s) to download")

    return files_to_download


def check_nas_http_server():
    """Check if NAS HTTP server is accessible

    Returns:
        bool: True if server is accessible, False otherwise
    """
    print(f"\n🌐 Checking NAS HTTP server...")
    print(f"   URL: http://{NAS_IP}:{HTTP_PORT}")

    if DRY_RUN:
        print(f"   [DRY RUN] Would check: http://{NAS_IP}:{HTTP_PORT}/sha256-list.csv")
        print(f"   [DRY RUN] Assuming NAS HTTP server is accessible")
        return True

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

    if DRY_RUN:
        # In dry-run mode, still download the CSV for comparison purposes
        # This allows us to show what files would be synced
        print(f"   [DRY RUN] Downloading NAS CSV for comparison (read-only operation)")
        # Fall through to actual download logic below

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
        if DRY_RUN:
            print(f"   [DRY RUN] Cannot proceed without NAS CSV for comparison")
        return None

    if DRY_RUN:
        print(f"   ✅ Downloaded to: {nas_csv_local} (for comparison only)")
    else:
        print(f"   ✅ Downloaded to: {nas_csv_local}")
    return str(nas_csv_local)




def download_files_from_nas(gpu_server, files, log_file=None, server_name=None):
    """Download files from NAS HTTP server to GPU server one by one

    Uses wget -O to download and overwrite existing files automatically.
    Uses dot format (--progress=dot:mega) for cleaner log output.
    After each successful download, updates sha256sum.csv on the GPU server.

    Args:
        gpu_server: Server spec in format user@hostname:/path
        files: List of filenames to download
        log_file: Optional file object to write logs to
        server_name: Server name for progress tracking

    Returns:
        int: Number of files successfully downloaded
    """
    if not files:
        log_print(log_file, "   ✅ No files to download")
        return 0

    # Extract hostname and path
    hostname, path = gpu_server.split(':')
    http_url = f"http://{NAS_IP}:{HTTP_PORT}"

    log_print(log_file, f"\n📥 Downloading {len(files)} file(s) from NAS...")
    log_print(log_file, f"   From: {http_url}")
    log_print(log_file, f"   To: {gpu_server}")

    if DRY_RUN:
        log_print(log_file, f"\n   [DRY RUN] Would download the following files:")
        for i, filename in enumerate(files, 1):
            file_url = f"{http_url}/{filename}"
            dest_path = f"{path}/{filename}"
            log_print(log_file, f"   [{i}/{len(files)}] {filename}")
            log_print(log_file, f"      From: {file_url}")
            log_print(log_file, f"      To:   {dest_path}")
            log_print(log_file, f"      [DRY RUN] Would update checksum after download")
        log_print(log_file, f"\n   [DRY RUN] Would download {len(files)} file(s)")
        return len(files)

    success_count = 0

    for i, filename in enumerate(files, 1):
        file_url = f"{http_url}/{filename}"
        dest_path = f"{path}/{filename}"

        # Update progress
        if server_name:
            update_progress(server_name, "Downloading", i-1, len(files))

        # Show progress
        log_print(log_file, f"   [{i}/{len(files)}] {filename}")

        # Download file using wget with dot format for cleaner logs
        # -O will overwrite existing file automatically (no need for rm)
        # --progress=dot:mega shows progress in dot format (cleaner for logs)
        wget_cmd = f'wget --progress=dot:mega "{file_url}" -O "{dest_path}"'

        # Stream output to log file if provided
        if log_file:
            result = subprocess.run(
                ['ssh', hostname, wget_cmd],
                text=True,
                stdout=log_file,
                stderr=log_file,
                timeout=3600  # 1 hour per file
            )
        else:
            result = subprocess.run(
                ['ssh', hostname, wget_cmd],
                text=True,
                timeout=3600  # 1 hour per file
            )

        if result.returncode == 0:
            success_count += 1

            # Calculate and update sha256-list.csv immediately after download
            log_print(log_file, f"      Updating checksum for {filename}...")
            checksum_cmd = f'cd "{path}" && ' \
                          f'checksum=$(sha256sum "{filename}" | cut -d" " -f1) && ' \
                          f'{{ echo "filename,sha256sum"; ' \
                          f'{{ cat sha256-list.csv 2>/dev/null | grep -v "^filename,sha256sum$" | grep -v "^{filename}," || true; }} ; ' \
                          f'echo "{filename},$checksum"; }} | ' \
                          f'{{ read header; echo "$header"; sort -t, -k1; }} > /tmp/sorted.csv && mv /tmp/sorted.csv sha256-list.csv'

            checksum_result = subprocess.run(
                ['ssh', hostname, checksum_cmd],
                text=True,
                capture_output=True,
                timeout=300  # 5 minutes for checksum
            )

            if checksum_result.returncode == 0:
                log_print(log_file, f"      ✅ Checksum updated")
            else:
                log_print(log_file, f"      ⚠️  Failed to update checksum: {checksum_result.stderr}")

            # Update progress after successful download
            if server_name:
                update_progress(server_name, "Downloading", success_count, len(files))
        else:
            log_print(log_file, f"   ⚠️  Failed: {filename}")
            if i <= 3:  # Show error for first few failures
                log_print(log_file, f"      Error: wget returned code {result.returncode}")

    log_print(log_file, f"   ✅ Downloaded {success_count}/{len(files)} file(s)")
    return success_count




def sync_single_server(gpu_server, nas_csv_local, log_file=None, server_name=None):
    """Sync a single GPU server with NAS

    Workflow:
    1. Update checksums on GPU server
    2. Compare checksums to get files to download
    3. Download files from NAS HTTP server
    4. Update checksums on GPU server again to verify

    Args:
        gpu_server: Server spec in format user@hostname:/path
        nas_csv_local: Local path to NAS CSV file (source of truth)
        log_file: Optional file object to write logs to
        server_name: Server name for progress tracking

    Returns:
        bool: True if sync successful, False otherwise
    """
    log_print(log_file, f"\n{'='*70}")
    log_print(log_file, f"Syncing: {gpu_server}")
    log_print(log_file, '='*70)

    try:
        # Step 1: Update checksums on GPU server
        if server_name:
            update_progress(server_name, "Checking checksums", 0, 0)
        if not log_file:
            print_step(1, 4, f"Update checksums on {gpu_server}")
        else:
            log_print(log_file, f"\n[Step 1/4] Update checksums on {gpu_server}")
        gpu_csv_local = update_gpu_server_checksums(gpu_server, log_file)

        # Step 2: Compare checksums to get files to download
        if server_name:
            update_progress(server_name, "Comparing checksums", 0, 0)
        if not log_file:
            print_step(2, 4, "Compare checksums")
        else:
            log_print(log_file, f"\n[Step 2/4] Compare checksums")
        files_to_download = get_files_to_download(gpu_csv_local, nas_csv_local, log_file)

        if not files_to_download:
            if server_name:
                update_progress(server_name, "Completed", 0, 0, status="completed")
            log_print(log_file, "\n✅ Server already in sync - no files to download")
            return True

        # Step 3: Download files from NAS
        if server_name:
            update_progress(server_name, "Downloading", 0, len(files_to_download))
        if not log_file:
            print_step(3, 4, "Download files from NAS")
        else:
            log_print(log_file, f"\n[Step 3/4] Download files from NAS")
        success_count = download_files_from_nas(gpu_server, files_to_download, log_file, server_name)

        if success_count < len(files_to_download):
            log_print(log_file, f"\n⚠️  Warning: Only {success_count}/{len(files_to_download)} files downloaded successfully")

        # Step 4: Update checksums on GPU server again to verify
        if server_name:
            update_progress(server_name, "Verifying checksums", success_count, len(files_to_download))
        if not log_file:
            print_step(4, 4, f"Verify checksums on {gpu_server}")
        else:
            log_print(log_file, f"\n[Step 4/4] Verify checksums on {gpu_server}")
        update_gpu_server_checksums(gpu_server, log_file)

        if server_name:
            update_progress(server_name, "Completed", success_count, len(files_to_download), status="completed")
        log_print(log_file, f"\n✅ Server sync completed: {success_count}/{len(files_to_download)} files downloaded")
        return True

    except Exception as e:
        if server_name:
            update_progress(server_name, "Failed", 0, 0, status="failed")
        log_print(log_file, f"\n❌ Error syncing {gpu_server}: {e}")
        import traceback
        if log_file:
            traceback.print_exc(file=log_file)
        else:
            traceback.print_exc()
        return False


def sync_server_with_logging(server, nas_csv_local, log_dir="."):
    """Wrapper to sync a single server with logging to file

    Args:
        server: Server spec in format user@hostname:/path
        nas_csv_local: Local path to NAS CSV file (source of truth)
        log_dir: Directory to write log files to

    Returns:
        bool: True if sync successful, False otherwise
    """
    # Extract hostname for log file naming
    hostname = server.split('@')[-1].split(':')[0]
    log_filename = f"{log_dir}/sync-{hostname}.log"

    # Open log file
    with open(log_filename, 'w') as log_file:
        # Write header to log
        log_file.write("="*70 + "\n")
        log_file.write(f"Sync Log for {server}\n")
        log_file.write(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write("="*70 + "\n\n")
        log_file.flush()

        # Sync the server
        success = sync_single_server(server, nas_csv_local, log_file, hostname)

        # Write footer to log
        log_file.write("\n" + "="*70 + "\n")
        log_file.write(f"Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Status: {'SUCCESS' if success else 'FAILED'}\n")
        log_file.write("="*70 + "\n")

    return success


def main():
    global DRY_RUN

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Multi-Server Model Sync with NAS HTTP Server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Perform actual sync (sequential)
  python3 sync_models_multi_server.py

  # Parallel sync (all servers at once)
  python3 sync_models_multi_server.py --parallel

  # Dry-run mode (show what would be done without making changes)
  python3 sync_models_multi_server.py --dry-run

  # Parallel dry-run
  python3 sync_models_multi_server.py --parallel --dry-run
        """
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without making any changes'
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Sync all servers in parallel (faster but more resource intensive)'
    )

    args = parser.parse_args()
    DRY_RUN = args.dry_run

    print("="*70)
    print("Multi-Server Model Sync with NAS HTTP Server")
    if DRY_RUN:
        print("[DRY RUN MODE - No changes will be made]")
    if args.parallel:
        print("[PARALLEL MODE - All servers synced simultaneously]")
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

        if args.parallel:
            # Parallel execution - sync all servers simultaneously
            print(f"\n🚀 Syncing {len(servers)} servers in parallel...")
            print(f"   Detailed logs will be written to sync-*.log files\n")

            # Display initial status
            for server in servers:
                hostname = server.split('@')[-1].split(':')[0]
                log_filename = f"sync-{hostname}.log"
                update_progress(hostname, "Starting", 0, 0)
                print(f"[{hostname:15}] Starting...{' '*25} | Log: {log_filename}")

            print()

            # Start monitoring thread
            stop_monitoring = threading.Event()

            def monitor_progress():
                """Monitor and display progress periodically"""
                while not stop_monitoring.is_set():
                    time.sleep(30)  # Update every 30 seconds
                    if not stop_monitoring.is_set():
                        display_progress_status()

            monitor_thread = threading.Thread(target=monitor_progress, daemon=True)
            monitor_thread.start()

            # Sync all servers in parallel
            with ThreadPoolExecutor(max_workers=len(servers)) as executor:
                # Submit all server sync tasks
                future_to_server = {
                    executor.submit(sync_server_with_logging, server, nas_csv_local, "."): server
                    for server in servers
                }

                # Process results as they complete
                for future in as_completed(future_to_server):
                    server = future_to_server[future]
                    hostname = server.split('@')[-1].split(':')[0]
                    try:
                        success = future.result()
                        results[server] = success

                        # Display completion message
                        with PROGRESS_LOCK:
                            data = PROGRESS_DATA.get(hostname, {})
                            files_synced = data.get("files_synced", 0)
                            total_files = data.get("total_files", 0)
                            if total_files > 0:
                                print(f"\n[{hostname}] ✅ Completed: {files_synced}/{total_files} files synced")
                            else:
                                print(f"\n[{hostname}] ✅ Completed: Already in sync")
                    except Exception as e:
                        print(f"\n[{hostname}] ❌ Exception: {e}")
                        results[server] = False

            # Stop monitoring thread
            stop_monitoring.set()
            monitor_thread.join(timeout=1)

            # Final status display
            display_progress_status()

        else:
            # Sequential execution - sync servers one at a time
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
