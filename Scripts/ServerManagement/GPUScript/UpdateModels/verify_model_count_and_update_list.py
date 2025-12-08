#!/usr/bin/env python3
"""
Verify that all GPU servers have the same number of models (.ckpt and .ckpt-tensordata).
Reads server list from gpu_servers.txt, compares model counts, and optionally
triggers control panel update.
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Paths derived from script location
SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parents[3]  # Scripts/ServerManagement/GPUScript/UpdateModels -> repo root
SERVERS_FILE = SCRIPT_DIR / "gpu_servers.txt"
MODEL_LIST_FILE = REPO_ROOT / "model-list"
MODELS_PATH = "/mnt/models/official-models/"
SSH_TIMEOUT = 30

# Control panel settings
CONTROL_PANEL_HOST = "100.80.251.87"
CONTROL_PANEL_PORT = "50002"


def parse_servers(file_path: Path) -> list[str]:
    """Parse server list from config file, returning user@hostname entries."""
    servers = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue
            # Extract user@hostname (ignore NAS override if present)
            server = line.split("|")[0]
            servers.append(server)
    return servers


def get_file_count(server: str, pattern: str) -> tuple[int | None, str]:
    """
    SSH into server and count files matching pattern.
    Returns (count, error_message). count is None if error occurred.
    """
    cmd = [
        "ssh",
        "-o", "ConnectTimeout=10",
        "-o", "StrictHostKeyChecking=no",
        "-o", "BatchMode=yes",
        server,
        f"ls -lrta {MODELS_PATH} 2>/dev/null | awk '/{pattern}$/ {{print $NF}}' | sort | wc -l"
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=SSH_TIMEOUT
        )
        if result.returncode != 0:
            return None, result.stderr.strip() or "SSH connection failed"

        count = int(result.stdout.strip())
        return count, ""
    except subprocess.TimeoutExpired:
        return None, "Connection timed out"
    except ValueError as e:
        return None, f"Failed to parse count: {e}"
    except Exception as e:
        return None, str(e)


def get_model_list(server: str) -> tuple[list[str] | None, str]:
    """
    SSH into server and get sorted list of .ckpt model names.
    Returns (model_list, error_message). model_list is None if error occurred.
    """
    cmd = [
        "ssh",
        "-o", "ConnectTimeout=10",
        "-o", "StrictHostKeyChecking=no",
        "-o", "BatchMode=yes",
        server,
        f"ls -lrta {MODELS_PATH} 2>/dev/null | awk '/\\.ckpt$/ {{print $NF}}' | sort"
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=SSH_TIMEOUT
        )
        if result.returncode != 0:
            return None, result.stderr.strip() or "SSH connection failed"

        models = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
        return models, ""
    except subprocess.TimeoutExpired:
        return None, "Connection timed out"
    except Exception as e:
        return None, str(e)


def update_control_panel(model_list_path: Path) -> tuple[bool, str]:
    """
    Run bazel command to update control panel with model list.
    Returns (success, output_or_error).
    """
    cmd = [
        "bazel", "run", "Apps:ProxyServerControlPanelCLI", "--",
        "-h", CONTROL_PANEL_HOST,
        "-p", CONTROL_PANEL_PORT,
        "update-model-list", str(model_list_path)
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            timeout=300  # 5 minutes for bazel build + run
        )
        output = result.stdout + result.stderr
        if result.returncode != 0:
            return False, output
        return True, output
    except subprocess.TimeoutExpired:
        return False, "Command timed out"
    except Exception as e:
        return False, str(e)


def verify_counts(servers: list[str]) -> tuple[dict[str, dict], dict[str, str]]:
    """
    Verify .ckpt and .ckpt-tensordata counts across all servers.
    Returns (results, errors) where results maps server -> {ckpt: count, tensordata: count}
    """
    results = {}
    errors = {}

    for server in servers:
        print(f"Checking {server}...", end=" ", flush=True)

        ckpt_count, ckpt_error = get_file_count(server, "\\.ckpt")
        if ckpt_error:
            errors[server] = ckpt_error
            print(f"ERROR: {ckpt_error}")
            continue

        tensordata_count, tensordata_error = get_file_count(server, "\\.ckpt-tensordata")
        if tensordata_error:
            errors[server] = tensordata_error
            print(f"ERROR: {tensordata_error}")
            continue

        results[server] = {
            "ckpt": ckpt_count,
            "tensordata": tensordata_count
        }
        print(f".ckpt: {ckpt_count}, .ckpt-tensordata: {tensordata_count}")

    return results, errors


def check_consistency(results: dict[str, dict]) -> tuple[bool, str]:
    """
    Check if all servers have consistent counts.
    Returns (is_consistent, summary_message).
    """
    if not results:
        return False, "No results to check"

    ckpt_counts = {s: r["ckpt"] for s, r in results.items()}
    tensordata_counts = {s: r["tensordata"] for s, r in results.items()}

    unique_ckpt = set(ckpt_counts.values())
    unique_tensordata = set(tensordata_counts.values())

    messages = []
    consistent = True

    if len(unique_ckpt) == 1:
        messages.append(f"All servers have {list(unique_ckpt)[0]} .ckpt files")
    else:
        consistent = False
        messages.append(".ckpt count mismatch:")
        count_to_servers = {}
        for server, count in ckpt_counts.items():
            count_to_servers.setdefault(count, []).append(server)
        for count, server_list in sorted(count_to_servers.items(), reverse=True):
            messages.append(f"  {count}: {', '.join(server_list)}")

    if len(unique_tensordata) == 1:
        messages.append(f"All servers have {list(unique_tensordata)[0]} .ckpt-tensordata files")
    else:
        consistent = False
        messages.append(".ckpt-tensordata count mismatch:")
        count_to_servers = {}
        for server, count in tensordata_counts.items():
            count_to_servers.setdefault(count, []).append(server)
        for count, server_list in sorted(count_to_servers.items(), reverse=True):
            messages.append(f"  {count}: {', '.join(server_list)}")

    return consistent, "\n".join(messages)


def main():
    parser = argparse.ArgumentParser(
        description="Verify model counts across GPU servers and optionally update control panel"
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Update control panel after successful verification"
    )
    parser.add_argument(
        "--force-update",
        action="store_true",
        help="Update control panel even if verification fails"
    )
    args = parser.parse_args()

    if not SERVERS_FILE.exists():
        print(f"Error: Server file not found: {SERVERS_FILE}")
        sys.exit(1)

    servers = parse_servers(SERVERS_FILE)
    if not servers:
        print("Error: No servers found in config file")
        sys.exit(1)

    print(f"Repository root: {REPO_ROOT}")
    print(f"Checking model counts on {len(servers)} servers...")
    print(f"Models path: {MODELS_PATH}")
    print("-" * 70)

    results, errors = verify_counts(servers)

    print("-" * 70)

    # Report errors
    if errors:
        print(f"\n⚠ {len(errors)} server(s) failed to connect:")
        for server, error in errors.items():
            print(f"  - {server}: {error}")

    if not results:
        print("\nNo successful connections. Cannot verify model counts.")
        sys.exit(1)

    # Check consistency
    consistent, summary = check_consistency(results)
    print(f"\n{'✓' if consistent else '✗'} Verification {'passed' if consistent else 'failed'}:")
    print(summary)

    # Determine if we should update
    should_update = args.force_update or (args.update and consistent and not errors)

    if args.update and not consistent:
        print("\n⚠ Skipping control panel update due to verification failure")
        print("  Use --force-update to update anyway")

    if args.update and errors and consistent:
        print("\n⚠ Skipping control panel update due to connection errors")
        print("  Use --force-update to update anyway")

    if should_update:
        print("\n" + "=" * 70)
        print("Updating control panel...")

        # Get model list from first successful server
        first_server = list(results.keys())[0]
        print(f"Fetching model list from {first_server}...")

        model_list, error = get_model_list(first_server)
        if error:
            print(f"Error fetching model list: {error}")
            sys.exit(1)

        # Save model list to file
        print(f"Saving {len(model_list)} models to {MODEL_LIST_FILE}...")
        with open(MODEL_LIST_FILE, "w") as f:
            f.write("\n".join(model_list) + "\n")

        # Run bazel command
        print(f"Running control panel update...")
        print(f"  Host: {CONTROL_PANEL_HOST}:{CONTROL_PANEL_PORT}")
        success, output = update_control_panel(MODEL_LIST_FILE)

        if success:
            print("✓ Control panel updated successfully")
        else:
            print(f"✗ Control panel update failed:\n{output}")
            sys.exit(1)

    if not consistent or errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
