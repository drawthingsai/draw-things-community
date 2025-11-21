#!/usr/bin/env python3
"""
Script to generate and compare SHA256 checksums for .ckpt-tensordata and .ckpt files.
"""

import os
import sys
import csv
import hashlib
import subprocess
from pathlib import Path


def parse_remote_path(path):
    """
    Parse a path that might be remote (user@host:/path) or local (/path).

    Returns:
        tuple: (is_remote, ssh_host, remote_path) or (False, None, local_path)
    """
    if ':' in path and '@' in path.split(':')[0]:
        # Format: user@host:/path
        ssh_part, remote_path = path.split(':', 1)
        return (True, ssh_part, remote_path)
    else:
        return (False, None, path)


def get_file_size(file_path, ssh_host=None, remote_base_path=None):
    """
    Get file size in bytes.

    Args:
        file_path: Local file path or just filename if remote
        ssh_host: SSH host (user@host) if checking remotely
        remote_base_path: Base path on remote system

    Returns:
        int: File size in bytes, or -1 if error occurred
    """
    try:
        if ssh_host:
            # Remote file - get size via SSH
            full_remote_path = f"{remote_base_path}/{file_path}"
            result = subprocess.run(
                ['ssh', ssh_host, f'stat -c %s "{full_remote_path}"'],
                capture_output=True,
                text=True,
                check=True
            )
            return int(result.stdout.strip())
        else:
            # Local file
            return os.path.getsize(file_path)
    except (subprocess.CalledProcessError, ValueError, OSError) as e:
        print(f"Error getting file size for {file_path}: {e}")
        return -1


def calculate_sha256(file_path, ssh_host=None, remote_base_path=None):
    """
    Calculate SHA256 checksum of a file using sha256sum command.

    Args:
        file_path: Local file path or just filename if remote
        ssh_host: SSH host (user@host) if calculating remotely
        remote_base_path: Base path on remote system

    Returns:
        str: SHA256 hash, or "Error" if calculation failed
    """
    try:
        if ssh_host:
            # Remote file - run sha256sum via SSH
            # Use forward slashes for remote paths (Linux/Unix)
            full_remote_path = f"{remote_base_path}/{file_path}"
            # Pass the entire command as a single string to SSH
            result = subprocess.run(
                ['ssh', ssh_host, f'sha256sum "{full_remote_path}"'],
                capture_output=True,
                text=True,
                check=True
            )
        else:
            # Local file
            result = subprocess.run(
                ['sha256sum', file_path],
                capture_output=True,
                text=True,
                check=True
            )
        # sha256sum output format: "<hash>  <filename>"
        sha256_hash = result.stdout.split()[0]
        return sha256_hash
    except subprocess.CalledProcessError as e:
        print(f"Error calculating SHA256 for {file_path}: {e}")
        stderr_msg = e.stderr if hasattr(e, 'stderr') and e.stderr else 'N/A'
        print(f"stderr: {stderr_msg}")
        return "Error"
    except Exception as e:
        print(f"Unexpected error for {file_path}: {e}")
        return "Error"


def remove_files(file_list, directory=None, ssh_host=None, remote_base_path=None):
    """
    Remove multiple files in batch (local or remote).

    Args:
        file_list: List of filenames to remove
        directory: Local directory path (for local files)
        ssh_host: SSH host (user@host) if removing remotely
        remote_base_path: Base path on remote system

    Returns:
        int: Number of files successfully removed
    """
    if not file_list:
        return 0

    removed_count = 0
    try:
        if ssh_host:
            # Remote files - build rm command with all files
            # Quote each filename to handle spaces/special chars
            files_to_remove = ' '.join([f'"{remote_base_path}/{f}"' for f in file_list])
            subprocess.run(
                ['ssh', ssh_host, f'rm -f {files_to_remove}'],
                capture_output=True,
                text=True,
                check=True
            )
            removed_count = len(file_list)
            for f in file_list:
                print(f"  ✓ Removed remote file: {f}")
        else:
            # Local files - remove each one
            for filename in file_list:
                file_path = os.path.join(directory, filename)
                try:
                    os.remove(file_path)
                    print(f"  ✓ Removed local file: {filename}")
                    removed_count += 1
                except OSError as e:
                    print(f"  ✗ Error removing {filename}: {e}")
        return removed_count
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Error removing remote files: {e}")
        return 0


def list_remote_files(ssh_host, remote_path):
    """
    List .ckpt-tensordata and .ckpt files in a remote directory via SSH.

    Args:
        ssh_host: SSH host (user@host)
        remote_path: Path on remote system

    Returns:
        list: Sorted list of filenames
    """
    try:
        # Use find command to list files
        result = subprocess.run(
            ['ssh', ssh_host, f'find "{remote_path}" -maxdepth 1 -type f \\( -name "*.ckpt-tensordata" -o -name "*.ckpt" \\) -printf "%f\\n"'],
            capture_output=True,
            text=True,
            check=True
        )
        files = [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]
        return sorted(files)
    except subprocess.CalledProcessError as e:
        print(f"Error listing remote files: {e}")
        return []


def update_csv_with_checksums(directory, csv_file, dry_run=False, ssh_host=None, remote_path=None, force=False):
    """
    Function 1: Update CSV file with SHA256 checksums.

    Args:
        directory: Directory containing .ckpt-tensordata and .ckpt files (local only)
        csv_file: CSV file path with columns: filename, sha256sum
        dry_run: If True, only print what would be done without writing files
        ssh_host: SSH host (user@host) if working with remote directory
        remote_path: Path on remote system
        force: If True, recalculate all checksums even if they already exist

    If sha256sum already exists for a file, skip it (unless force=True). Otherwise calculate and fill it.
    """
    display_path = f"{ssh_host}:{remote_path}" if ssh_host else directory
    print(f"\n=== Processing directory: {display_path} ===")
    if dry_run:
        print("(DRY-RUN MODE: No files will be written)")
    if force:
        print("(FORCE MODE: Recalculating all checksums)")

    # Read existing CSV data
    checksums = {}
    csv_entry_count = 0
    if os.path.exists(csv_file):
        with open(csv_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row.get('filename', '')
                sha256 = row.get('sha256sum', '').strip()
                checksums[filename] = sha256
                csv_entry_count += 1

    # Get all .ckpt-tensordata and .ckpt files in directory
    if ssh_host:
        # Remote directory
        tensordata_files = list_remote_files(ssh_host, remote_path)
    else:
        # Local directory
        tensordata_files = sorted([
            f for f in os.listdir(directory)
            if (f.endswith('.ckpt-tensordata') or f.endswith('.ckpt')) and os.path.isfile(os.path.join(directory, f))
        ])

    # Print summary
    print(f"\nSummary:")
    print(f"  CSV entries (existing):     {csv_entry_count}")
    print(f"  Files in directory:         {len(tensordata_files)}")

    # Count files that need processing
    if force:
        # In force mode, all files need processing (clear existing checksums)
        for filename in tensordata_files:
            checksums[filename] = ''
        new_files = tensordata_files
        print(f"  Files to process (force):   {len(new_files)}")
    else:
        new_files = [f for f in tensordata_files if f not in checksums or not checksums[f]]
        print(f"  New files to process:       {len(new_files)}")
        print(f"  Files with checksums:       {csv_entry_count - len(new_files) if csv_entry_count >= len(new_files) else 0}")

        # Ensure all files are in the dictionary
        for filename in tensordata_files:
            if filename not in checksums:
                checksums[filename] = ''

    if dry_run:
        # Dry-run mode: just print what would be done
        files_to_process = [f for f in tensordata_files if f not in checksums or not checksums[f]]
        if files_to_process:
            print(f"\nFiles that would be processed:")
            for filename in files_to_process:
                print(f"  - {filename}")
            print(f"\nTotal: {len(files_to_process)} file(s) would be processed")
        else:
            print("  (No files to process)")

        if not force:
            existing_files = [f for f in tensordata_files if f in checksums and checksums[f]]
            if existing_files:
                print(f"\nFiles with existing checksums (would be skipped):")
                for filename in existing_files:
                    print(f"  - {filename}")
        return checksums

    # Calculate missing checksums and identify problematic files
    updated = False
    files_to_remove = []  # Track corrupted and zero-size files

    for filename in tensordata_files:
        if not checksums[filename]:  # Empty or missing checksum
            # First check file size
            if ssh_host:
                file_size = get_file_size(filename, ssh_host, remote_path)
            else:
                file_path = os.path.join(directory, filename)
                file_size = get_file_size(file_path)

            # Mark zero-size files for removal
            if file_size == 0:
                print(f"⚠ Zero-size file detected: {filename}")
                files_to_remove.append(filename)
                checksums[filename] = "ZeroSize"
                continue
            elif file_size == -1:
                print(f"⚠ Error getting file size: {filename}")
                files_to_remove.append(filename)
                checksums[filename] = "Error"
                continue

            # Calculate checksum
            if ssh_host:
                # Remote file
                print(f"Calculating SHA256 for: {filename} (remote)")
                sha256 = calculate_sha256(filename, ssh_host, remote_path)
            else:
                # Local file
                file_path = os.path.join(directory, filename)
                print(f"Calculating SHA256 for: {filename}")
                sha256 = calculate_sha256(file_path)

            # Mark files with checksum errors for removal
            if sha256 == "Error":
                print(f"⚠ Checksum calculation failed: {filename}")
                files_to_remove.append(filename)

            checksums[filename] = sha256
            updated = True
            print(f"  -> {sha256}")

            # In local mode, update CSV immediately after each checksum
            if not ssh_host:
                with open(csv_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['filename', 'sha256sum'])
                    for fn in sorted(checksums.keys()):
                        writer.writerow([fn, checksums[fn]])
        else:
            if force:
                # Force mode: recalculate even if checksum exists
                print(f"Recalculating SHA256 for: {filename} (force mode)")

                # First check file size
                if ssh_host:
                    file_size = get_file_size(filename, ssh_host, remote_path)
                else:
                    file_path = os.path.join(directory, filename)
                    file_size = get_file_size(file_path)

                # Mark zero-size files for removal
                if file_size == 0:
                    print(f"⚠ Zero-size file detected: {filename}")
                    files_to_remove.append(filename)
                    checksums[filename] = "ZeroSize"
                    continue
                elif file_size == -1:
                    print(f"⚠ Error getting file size: {filename}")
                    files_to_remove.append(filename)
                    checksums[filename] = "Error"
                    continue

                # Calculate checksum
                if ssh_host:
                    sha256 = calculate_sha256(filename, ssh_host, remote_path)
                else:
                    file_path = os.path.join(directory, filename)
                    sha256 = calculate_sha256(file_path)

                # Mark files with checksum errors for removal
                if sha256 == "Error":
                    print(f"⚠ Checksum calculation failed: {filename}")
                    files_to_remove.append(filename)

                checksums[filename] = sha256
                updated = True
                print(f"  -> {sha256}")

                # In local mode, update CSV immediately after each checksum
                if not ssh_host:
                    with open(csv_file, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['filename', 'sha256sum'])
                        for fn in sorted(checksums.keys()):
                            writer.writerow([fn, checksums[fn]])
            else:
                print(f"Skipping {filename} (checksum already exists)")

    # Remove corrupted and zero-size files
    if files_to_remove:
        print(f"\n=== Cleaning up {len(files_to_remove)} corrupted/zero-size file(s) ===")
        for f in files_to_remove:
            print(f"  - {f} (reason: {checksums[f]})")

        # Remove files from directory
        removed_count = remove_files(files_to_remove, directory, ssh_host, remote_path)
        print(f"\nRemoved {removed_count} file(s) from directory")

        # Remove from checksums dictionary (won't be written to CSV)
        for f in files_to_remove:
            del checksums[f]

        print(f"Removed {len(files_to_remove)} file(s) from CSV records")

    # Write back to CSV (only valid files)
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'sha256sum'])
        for filename in sorted(checksums.keys()):
            writer.writerow([filename, checksums[filename]])

    if updated:
        print(f"Updated CSV file: {csv_file}")
    else:
        print(f"No updates needed for: {csv_file}")

    return checksums


def compare_checksums(csv_file1, csv_file2, verbose=False):
    """
    Function 2: Compare two CSV files and print differences.

    Args:
        csv_file1: First CSV file (typically from server being checked)
        csv_file2: Second CSV file (source of truth)
        verbose: If True, show detailed error messages. If False (default),
                 only list filenames to download

    Prints files that:
    - Have missing SHA256 in either file
    - Have different SHA256 values between the two files
    """
    # Read first CSV
    checksums1 = {}
    with open(csv_file1, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row.get('filename', '')
            sha256 = row.get('sha256sum', '').strip()
            checksums1[filename] = sha256

    # Read second CSV
    checksums2 = {}
    with open(csv_file2, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row.get('filename', '')
            sha256 = row.get('sha256sum', '').strip()
            checksums2[filename] = sha256

    # Get all unique filenames
    all_files = sorted(set(checksums1.keys()) | set(checksums2.keys()))

    files_to_download = []
    files_with_mismatch = []
    missing_in_source = []

    for filename in all_files:
        sha1 = checksums1.get(filename, '')
        sha2 = checksums2.get(filename, '')

        # Check for missing checksums
        if not sha1 or not sha2:
            if not sha1 and not sha2:
                # Missing in both - skip
                pass
            elif not sha1:
                # Missing in first directory - need to download from source
                files_to_download.append(filename)
            else:
                # Missing in source of truth - file exists locally but not in source
                missing_in_source.append(filename)
        # Check for different checksums
        elif sha1 != sha2:
            files_with_mismatch.append(filename)

    # Output results
    differences_found = bool(files_to_download or files_with_mismatch or missing_in_source)

    if verbose:
        # Detailed output mode
        print(f"\n=== Comparing checksums ===")

        if files_to_download:
            print(f"\n❌ Files missing in first directory ({len(files_to_download)}):")
            for f in files_to_download:
                print(f"  {f}")

        if files_with_mismatch:
            print(f"\n❌ Files with checksum mismatch ({len(files_with_mismatch)}):")
            for f in files_with_mismatch:
                sha1 = checksums1.get(f, '')
                sha2 = checksums2.get(f, '')
                print(f"  {f}")
                print(f"    Dir1: {sha1}")
                print(f"    Dir2: {sha2}")

        if missing_in_source:
            print(f"\n⚠ Files in first directory but not in source ({len(missing_in_source)}):")
            for f in missing_in_source:
                print(f"  {f}")

        if not differences_found:
            print("✓ All checksums match!")
    else:
        # Light output mode (default) - just list files to download
        all_needed_files = files_to_download + files_with_mismatch

        if all_needed_files:
            for f in all_needed_files:
                print(f)
        else:
            # No output if everything matches (clean for scripting)
            pass

    return differences_found


def download_remote_csv(ssh_host, remote_path, local_csv_file):
    """
    Download CSV file from remote host if it exists.

    Returns:
        bool: True if file was downloaded, False otherwise
    """
    remote_csv = f"{remote_path}/sha256-list.csv"
    try:
        # Check if remote CSV exists
        result = subprocess.run(
            ['ssh', ssh_host, f'test -f "{remote_csv}" && echo "exists"'],
            capture_output=True,
            text=True,
            check=False
        )
        if result.stdout.strip() == "exists":
            print(f"Downloading existing CSV from remote: {remote_csv}")
            # Use scp to download the file
            # Don't use quotes around remote path when passing as list argument
            subprocess.run(
                ['scp', f'{ssh_host}:{remote_csv}', local_csv_file],
                check=True
            )
            print(f"Downloaded to: {local_csv_file}")
            return True
        else:
            print(f"No existing CSV found on remote host")
            return False
    except subprocess.CalledProcessError as e:
        print(f"Error downloading remote CSV: {e}")
        return False


def upload_csv_to_remote(ssh_host, remote_path, local_csv_file):
    """
    Upload CSV file to remote host.

    Args:
        ssh_host: SSH host (user@host)
        remote_path: Directory path on remote system
        local_csv_file: Local CSV file path

    Returns:
        bool: True if file was uploaded successfully, False otherwise
    """
    remote_csv = f"{remote_path}/sha256-list.csv"
    try:
        print(f"Uploading CSV to remote: {remote_csv}")
        subprocess.run(
            ['scp', local_csv_file, f'{ssh_host}:{remote_csv}'],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"  ✓ Uploaded: {local_csv_file} -> {ssh_host}:{remote_csv}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Error uploading CSV to remote: {e}")
        if hasattr(e, 'stderr') and e.stderr:
            print(f"  stderr: {e.stderr}")
        return False


def initialize_csv_from_directory(directory, csv_file, dry_run=False, ssh_host=None, remote_path=None):
    """Create CSV file with filenames if it doesn't exist."""
    if not os.path.exists(csv_file):
        if ssh_host:
            # Remote directory
            tensordata_files = list_remote_files(ssh_host, remote_path)
        else:
            # Local directory
            tensordata_files = sorted([
                f for f in os.listdir(directory)
                if (f.endswith('.ckpt-tensordata') or f.endswith('.ckpt')) and os.path.isfile(os.path.join(directory, f))
            ])

        if dry_run:
            print(f"Would create CSV file: {csv_file}")
            print(f"Would add {len(tensordata_files)} file(s) to CSV:")
            for filename in tensordata_files:
                print(f"  - {filename}")
            return

        print(f"Creating new CSV file: {csv_file}")
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['filename', 'sha256sum'])
            for filename in tensordata_files:
                writer.writerow([filename, ''])

        print(f"Created CSV with {len(tensordata_files)} files")


def main():
    # Check for flags
    dry_run = '--dry-run' in sys.argv
    verbose = '--verbose' in sys.argv or '-v' in sys.argv
    force = '--force' in sys.argv
    args = [arg for arg in sys.argv[1:] if arg not in ['--dry-run', '--verbose', '-v', '--force']]

    if len(args) < 1 or len(args) > 2:
        print("Usage: python compare_checksums.py [--dry-run] [--verbose|-v] [--force] <directory|csv1> [csv2]")
        print()
        print("Options:")
        print("  --dry-run: Print what would be done without writing files or calculating checksums")
        print("  --verbose, -v: Show detailed output in CSV comparison mode")
        print("  --force: Recalculate all checksums even if they already exist (single directory mode only)")
        print()
        print("Modes:")
        print("  1. Single directory mode: Creates and fills sha256-list.csv")
        print("     - Identifies and removes corrupted files (checksum errors)")
        print("     - Identifies and removes zero-size files")
        print("     - Cleans up CSV to match valid files only")
        print("     Local:  ./compare_checksums.py /path/to/dir1")
        print("     Remote: ./compare_checksums.py root@host:/path/to/dir1")
        print("             (Downloads sha256-list.csv from remote if exists,")
        print("              updates locally, removes bad files, uploads back)")
        print()
        print("  2. CSV comparison mode: Compare two CSV files")
        print("     Example: ./compare_checksums.py csv1 csv2")
        print("     Output (default): Simple list of files to download (one per line)")
        print("     Output (--verbose): Detailed differences with categories")
        print("     Note: csv2 is treated as the source of truth")
        sys.exit(1)

    first_arg = args[0]

    # Check if second argument is provided
    if len(args) == 2:
        second_arg = args[1]

        # CSV comparison mode - both arguments should be CSV files
        if first_arg.endswith('.csv') and second_arg.endswith('.csv'):
            if force:
                print("Warning: --force flag is ignored in CSV comparison mode")

            if not os.path.isfile(first_arg):
                print(f"Error: {first_arg} is not a valid file")
                sys.exit(1)
            if not os.path.isfile(second_arg):
                print(f"Error: {second_arg} is not a valid file")
                sys.exit(1)

            if verbose:
                print(f"CSV comparison mode: Comparing {first_arg} against source of truth: {second_arg}")

            # Function 2: Compare the two CSVs
            has_differences = compare_checksums(first_arg, second_arg, verbose)
            # Exit with appropriate code
            sys.exit(1 if has_differences else 0)
        else:
            print("Error: For CSV comparison mode, both arguments must be CSV files")
            sys.exit(1)
    else:
        # Single directory mode (local or remote)
        dir_arg = first_arg

        # Parse the directory argument (could be remote)
        is_remote, ssh_host, dir_path = parse_remote_path(dir_arg)

        if is_remote:
            # Remote directory mode
            print(f"Remote directory mode: {ssh_host}:{dir_path}")

            # Create a local CSV file based on the remote hostname
            # Extract hostname from ssh_host (user@hostname -> hostname)
            hostname = ssh_host.split('@')[-1] if '@' in ssh_host else ssh_host
            csv_file1 = f"sha256-list-{hostname}.csv"

            # Try to download existing CSV from remote host first
            if not dry_run:
                download_remote_csv(ssh_host, dir_path, csv_file1)

            # Initialize CSV file if it doesn't exist
            initialize_csv_from_directory(None, csv_file1, dry_run, ssh_host, dir_path)

            # Function 1: Update CSV with checksums (also cleans up corrupted files)
            update_csv_with_checksums(None, csv_file1, dry_run, ssh_host, dir_path, force)

            # Upload the updated CSV back to remote
            if not dry_run:
                upload_csv_to_remote(ssh_host, dir_path, csv_file1)
        else:
            # Local directory mode
            dir1 = dir_path

            # Validate directory
            if not os.path.isdir(dir1):
                print(f"Error: {dir1} is not a valid directory")
                sys.exit(1)

            csv_file1 = os.path.join(dir1, 'sha256-list.csv')

            # Initialize CSV file if it doesn't exist
            initialize_csv_from_directory(dir1, csv_file1, dry_run)

            # Function 1: Update CSV with checksums
            update_csv_with_checksums(dir1, csv_file1, dry_run, force=force)

        if dry_run:
            print("\n✓ Dry-run completed (no files modified)")
        else:
            print("\n✓ Single directory mode completed successfully")
        sys.exit(0)


if __name__ == '__main__':
    main()
