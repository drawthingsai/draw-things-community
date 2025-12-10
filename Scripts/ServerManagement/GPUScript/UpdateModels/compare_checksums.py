#!/usr/bin/env python3
"""
Script to generate and compare checksums for .ckpt-tensordata and .ckpt files.

Supports three levels of integrity checking:
  L1: File size comparison (fastest)
  L2: 8K hash - SHA256 of first 4KB + last 4KB (fast, good for detecting corruption)
  L3: Full SHA256 hash (slowest but most thorough)

CSV format: filename, sha256sum, 8k_sha256sum, filesize
"""

import os
import sys
import csv
import hashlib
import subprocess
from pathlib import Path


# CSV column names
CSV_COLUMNS = ['filename', 'sha256sum', '8k_sha256sum', 'filesize']


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


def calculate_8k_hash(file_path, ssh_host=None, remote_base_path=None):
    """
    Calculate SHA256 hash of first 4KB + last 4KB of a file.

    Args:
        file_path: Local file path or just filename if remote
        ssh_host: SSH host (user@host) if calculating remotely
        remote_base_path: Base path on remote system

    Returns:
        str: SHA256 hash of 8K sample, or "Error" if calculation failed
    """
    try:
        if ssh_host:
            # Remote file - use dd and sha256sum via SSH
            full_remote_path = f"{remote_base_path}/{file_path}"
            # Get file size first
            size_result = subprocess.run(
                ['ssh', ssh_host, f'stat -c %s "{full_remote_path}"'],
                capture_output=True,
                text=True,
                check=True
            )
            file_size = int(size_result.stdout.strip())

            if file_size <= 8192:
                # File is 8KB or smaller, just hash the whole file
                result = subprocess.run(
                    ['ssh', ssh_host, f'sha256sum "{full_remote_path}"'],
                    capture_output=True,
                    text=True,
                    check=True
                )
                return result.stdout.split()[0]

            # Calculate offset for last 4KB
            last_4k_offset = file_size - 4096

            # Read first 4KB and last 4KB, concatenate and hash
            # Using a bash script to combine both reads
            cmd = f'''
                (dd if="{full_remote_path}" bs=4096 count=1 2>/dev/null; \
                 dd if="{full_remote_path}" bs=1 skip={last_4k_offset} count=4096 2>/dev/null) | sha256sum
            '''
            result = subprocess.run(
                ['ssh', ssh_host, cmd],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.split()[0]
        else:
            # Local file
            file_size = os.path.getsize(file_path)

            if file_size <= 8192:
                # File is 8KB or smaller, just hash the whole file
                with open(file_path, 'rb') as f:
                    return hashlib.sha256(f.read()).hexdigest()

            # Read first 4KB and last 4KB
            with open(file_path, 'rb') as f:
                first_4k = f.read(4096)
                f.seek(-4096, 2)  # Seek to 4KB before end
                last_4k = f.read(4096)

            # Combine and hash
            combined = first_4k + last_4k
            return hashlib.sha256(combined).hexdigest()

    except (subprocess.CalledProcessError, OSError, ValueError) as e:
        print(f"Error calculating 8K hash for {file_path}: {e}")
        return "Error"


def calculate_sha256(file_path, ssh_host=None, remote_base_path=None):
    """
    Calculate full SHA256 checksum of a file using sha256sum command.

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
            full_remote_path = f"{remote_base_path}/{file_path}"
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
            files_to_remove = ' '.join([f'"{remote_base_path}/{f}"' for f in file_list])
            subprocess.run(
                ['ssh', ssh_host, f'rm -f {files_to_remove}'],
                capture_output=True,
                text=True,
                check=True
            )
            removed_count = len(file_list)
            for f in file_list:
                print(f"  Removed remote file: {f}")
        else:
            # Local files - remove each one
            for filename in file_list:
                file_path = os.path.join(directory, filename)
                try:
                    os.remove(file_path)
                    print(f"  Removed local file: {filename}")
                    removed_count += 1
                except OSError as e:
                    print(f"  Error removing {filename}: {e}")
        return removed_count
    except subprocess.CalledProcessError as e:
        print(f"  Error removing remote files: {e}")
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


def read_csv(csv_file):
    """
    Read CSV file into a dictionary.

    Returns:
        dict: {filename: {'sha256sum': str, '8k_sha256sum': str, 'filesize': int}}
    """
    checksums = {}
    if os.path.exists(csv_file):
        with open(csv_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                reader.fieldnames = [name.strip() for name in reader.fieldnames]
            for row in reader:
                if not row:
                    continue
                filename = (row.get('filename', '') or '').strip()
                if not filename:
                    continue

                checksums[filename] = {
                    'sha256sum': (row.get('sha256sum', '') or '').strip(),
                    '8k_sha256sum': (row.get('8k_sha256sum', '') or '').strip(),
                    'filesize': row.get('filesize', '').strip() if row.get('filesize') else ''
                }
                # Convert filesize to int if present
                if checksums[filename]['filesize']:
                    try:
                        checksums[filename]['filesize'] = int(checksums[filename]['filesize'])
                    except ValueError:
                        checksums[filename]['filesize'] = ''
    return checksums


def write_csv(csv_file, checksums):
    """
    Write checksums dictionary to CSV file.

    Args:
        csv_file: Path to CSV file
        checksums: dict {filename: {'sha256sum': str, '8k_sha256sum': str, 'filesize': int}}
    """
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        for filename in sorted(checksums.keys()):
            data = checksums[filename]
            filesize = data.get('filesize', '')
            if isinstance(filesize, int):
                filesize = str(filesize)
            writer.writerow([
                filename,
                data.get('sha256sum', ''),
                data.get('8k_sha256sum', ''),
                filesize
            ])


def update_csv_with_level(directory, csv_file, level, dry_run=False, ssh_host=None, remote_path=None, force=False, specific_files=None):
    """
    Update CSV file with checksums at the specified level.

    Args:
        directory: Directory containing files (local only)
        csv_file: CSV file path
        level: Integrity level ('L1', 'L2', or 'L3')
        dry_run: If True, only print what would be done
        ssh_host: SSH host (user@host) if working with remote directory
        remote_path: Path on remote system
        force: If True, recalculate even if value already exists
        specific_files: List of specific filenames to update (if None, process all files)
    """
    display_path = f"{ssh_host}:{remote_path}" if ssh_host else directory
    print(f"\n=== Processing directory: {display_path} ===")
    print(f"Level: {level}")
    if dry_run:
        print("(DRY-RUN MODE: No files will be written)")
    if force:
        print("(FORCE MODE: Recalculating all values)")
    if specific_files:
        print(f"(SPECIFIC FILES MODE: Processing {len(specific_files)} file(s))")

    # Read existing CSV data
    checksums = read_csv(csv_file)
    csv_entry_count = len(checksums)

    # Get all .ckpt-tensordata and .ckpt files in directory
    if ssh_host:
        tensordata_files = list_remote_files(ssh_host, remote_path)
    else:
        tensordata_files = sorted([
            f for f in os.listdir(directory)
            if (f.endswith('.ckpt-tensordata') or f.endswith('.ckpt')) and os.path.isfile(os.path.join(directory, f))
        ])

    # If specific files are requested, filter to only those files
    if specific_files:
        # Validate that specified files exist in the directory
        missing_files = [f for f in specific_files if f not in tensordata_files]
        if missing_files:
            print(f"\nWarning: The following files were not found in directory:")
            for f in missing_files:
                print(f"  - {f}")
        # Filter to only the specified files that exist
        tensordata_files = [f for f in specific_files if f in tensordata_files]
        if not tensordata_files:
            print("Error: None of the specified files were found in the directory")
            return checksums

    # Print summary
    print(f"\nSummary:")
    print(f"  CSV entries (existing):     {csv_entry_count}")
    print(f"  Files to consider:          {len(tensordata_files)}")

    # Ensure all files are in the dictionary
    for filename in tensordata_files:
        if filename not in checksums:
            checksums[filename] = {'sha256sum': '', '8k_sha256sum': '', 'filesize': ''}

    # Determine which field to update based on level
    if level == 'L1':
        field = 'filesize'
    elif level == 'L2':
        field = '8k_sha256sum'
    else:  # L3
        field = 'sha256sum'

    # Count files that need processing
    if force:
        files_to_process = tensordata_files
    else:
        files_to_process = [f for f in tensordata_files if not checksums[f].get(field)]

    print(f"  Files to process ({field}): {len(files_to_process)}")

    if dry_run:
        if files_to_process:
            print(f"\nFiles that would be processed:")
            for filename in files_to_process[:20]:  # Show first 20
                print(f"  - {filename}")
            if len(files_to_process) > 20:
                print(f"  ... and {len(files_to_process) - 20} more")
        else:
            print("  (No files to process)")
        return checksums

    # Process files
    files_to_remove = []
    updated = False

    for i, filename in enumerate(files_to_process):
        if ssh_host:
            file_ref = filename
        else:
            file_ref = os.path.join(directory, filename)

        # Progress indicator
        print(f"[{i+1}/{len(files_to_process)}] Processing: {filename}")

        if level == 'L1':
            # Get file size
            if ssh_host:
                value = get_file_size(filename, ssh_host, remote_path)
            else:
                value = get_file_size(file_ref)

            if value == 0:
                print(f"  Zero-size file detected")
                files_to_remove.append(filename)
                checksums[filename]['filesize'] = 'ZeroSize'
                continue
            elif value == -1:
                print(f"  Error getting file size")
                files_to_remove.append(filename)
                checksums[filename]['filesize'] = 'Error'
                continue

            checksums[filename]['filesize'] = value
            print(f"  -> {value} bytes")

        elif level == 'L2':
            # Calculate 8K hash
            if ssh_host:
                value = calculate_8k_hash(filename, ssh_host, remote_path)
            else:
                value = calculate_8k_hash(file_ref)

            if value == "Error":
                print(f"  8K hash calculation failed")
                files_to_remove.append(filename)

            checksums[filename]['8k_sha256sum'] = value
            print(f"  -> {value[:16]}..." if len(value) > 16 else f"  -> {value}")

        else:  # L3
            # Calculate full SHA256
            if ssh_host:
                value = calculate_sha256(filename, ssh_host, remote_path)
            else:
                value = calculate_sha256(file_ref)

            if value == "Error":
                print(f"  SHA256 calculation failed")
                files_to_remove.append(filename)

            checksums[filename]['sha256sum'] = value
            print(f"  -> {value[:16]}..." if len(value) > 16 else f"  -> {value}")

        updated = True

        # Save CSV after each file (for recovery)
        if not ssh_host:
            write_csv(csv_file, checksums)

    # Remove corrupted files
    if files_to_remove:
        print(f"\n=== Cleaning up {len(files_to_remove)} corrupted/zero-size file(s) ===")
        for f in files_to_remove:
            print(f"  - {f}")

        removed_count = remove_files(files_to_remove, directory, ssh_host, remote_path)
        print(f"\nRemoved {removed_count} file(s) from directory")

        for f in files_to_remove:
            del checksums[f]

    # Write final CSV
    write_csv(csv_file, checksums)

    if updated:
        print(f"Updated CSV file: {csv_file}")
    else:
        print(f"No updates needed for: {csv_file}")

    return checksums


def compare_checksums_all_levels(csv_file1, csv_file2, verbose=False):
    """
    Compare two CSV files at all levels (L1, L2, L3).

    Args:
        csv_file1: First CSV file (typically from server being checked)
        csv_file2: Second CSV file (source of truth)
        verbose: If True, show detailed output

    Returns:
        bool: True if differences found at any level
    """
    checksums1 = read_csv(csv_file1)
    checksums2 = read_csv(csv_file2)

    levels_config = [
        ('L1', 'filesize', 'file size'),
        ('L2', '8k_sha256sum', '8K hash'),
        ('L3', 'sha256sum', 'SHA256'),
    ]

    all_files = sorted(set(checksums1.keys()) | set(checksums2.keys()))
    all_diff_files = set()
    level_results = {}

    for level_name, field, field_label in levels_config:
        files_to_download = []
        files_with_mismatch = []
        missing_in_source = []

        for filename in all_files:
            data1 = checksums1.get(filename, {})
            data2 = checksums2.get(filename, {})

            val1 = data1.get(field, '')
            val2 = data2.get(field, '')

            val1_str = str(val1) if val1 else ''
            val2_str = str(val2) if val2 else ''

            if not val1_str and not val2_str:
                pass  # Field missing in both, skip
            elif not val1_str:
                files_to_download.append(filename)
            elif not val2_str:
                missing_in_source.append(filename)
            elif val1_str != val2_str:
                files_with_mismatch.append((filename, val1_str, val2_str))

        diff_files = set(files_to_download + [f for f, _, _ in files_with_mismatch])
        all_diff_files.update(diff_files)
        level_results[level_name] = {
            'field_label': field_label,
            'to_download': files_to_download,
            'mismatch': files_with_mismatch,
            'missing_in_source': missing_in_source,
            'diff_count': len(diff_files)
        }

    differences_found = bool(all_diff_files)

    if verbose:
        print(f"\n=== Comparing at ALL levels ===")
        print(f"CSV 1: {csv_file1}")
        print(f"CSV 2: {csv_file2} (source of truth)")

        for level_name, field, field_label in levels_config:
            result = level_results[level_name]
            print(f"\n--- {level_name} ({field_label}) ---")

            if result['to_download']:
                print(f"  Missing in first: {len(result['to_download'])} file(s)")
            if result['mismatch']:
                print(f"  Mismatch: {len(result['mismatch'])} file(s)")
                for f, v1, v2 in result['mismatch'][:5]:
                    print(f"    {f}: {v1} != {v2}")
                if len(result['mismatch']) > 5:
                    print(f"    ... and {len(result['mismatch']) - 5} more")
            if result['missing_in_source']:
                print(f"  Not in source: {len(result['missing_in_source'])} file(s)")
            if result['diff_count'] == 0:
                print(f"  All {field_label} values match!")

        print(f"\n=== Summary ===")
        print(f"Total files with differences: {len(all_diff_files)}")
    else:
        # Light output mode - just list unique files with differences
        for f in sorted(all_diff_files):
            print(f)

    return differences_found


def compare_checksums(csv_file1, csv_file2, level='L3', verbose=False):
    """
    Compare two CSV files at the specified level.

    Args:
        csv_file1: First CSV file (typically from server being checked)
        csv_file2: Second CSV file (source of truth)
        level: Comparison level ('L1', 'L2', 'L3', or 'ALL')
        verbose: If True, show detailed output

    Returns:
        bool: True if differences found
    """
    # Handle ALL level - compare at all levels
    if level == 'ALL':
        return compare_checksums_all_levels(csv_file1, csv_file2, verbose)

    checksums1 = read_csv(csv_file1)
    checksums2 = read_csv(csv_file2)

    # Determine which field to compare based on level
    if level == 'L1':
        field = 'filesize'
        field_label = 'file size'
    elif level == 'L2':
        field = '8k_sha256sum'
        field_label = '8K hash'
    else:  # L3
        field = 'sha256sum'
        field_label = 'SHA256'

    # Get all unique filenames
    all_files = sorted(set(checksums1.keys()) | set(checksums2.keys()))

    files_to_download = []
    files_with_mismatch = []
    missing_in_source = []
    missing_field_in_both = []

    for filename in all_files:
        data1 = checksums1.get(filename, {})
        data2 = checksums2.get(filename, {})

        val1 = data1.get(field, '')
        val2 = data2.get(field, '')

        # Convert to string for comparison
        val1_str = str(val1) if val1 else ''
        val2_str = str(val2) if val2 else ''

        if not val1_str and not val2_str:
            # Field missing in both
            missing_field_in_both.append(filename)
        elif not val1_str:
            # Missing in first file - need to download from source
            files_to_download.append(filename)
        elif not val2_str:
            # Missing in source of truth
            missing_in_source.append(filename)
        elif val1_str != val2_str:
            # Values don't match
            files_with_mismatch.append((filename, val1_str, val2_str))

    # Output results
    differences_found = bool(files_to_download or files_with_mismatch or missing_in_source)

    if verbose:
        print(f"\n=== Comparing at level {level} ({field_label}) ===")
        print(f"CSV 1: {csv_file1}")
        print(f"CSV 2: {csv_file2} (source of truth)")

        if missing_field_in_both:
            print(f"\n-- {field_label} not computed in either file ({len(missing_field_in_both)}):")
            for f in missing_field_in_both[:10]:
                print(f"  {f}")
            if len(missing_field_in_both) > 10:
                print(f"  ... and {len(missing_field_in_both) - 10} more")

        if files_to_download:
            print(f"\n-- Files missing in first directory ({len(files_to_download)}):")
            for f in files_to_download:
                print(f"  {f}")

        if files_with_mismatch:
            print(f"\n-- Files with {field_label} mismatch ({len(files_with_mismatch)}):")
            for f, v1, v2 in files_with_mismatch:
                print(f"  {f}")
                print(f"    Dir1: {v1}")
                print(f"    Dir2: {v2}")

        if missing_in_source:
            print(f"\n-- Files in first directory but not in source ({len(missing_in_source)}):")
            for f in missing_in_source:
                print(f"  {f}")

        if not differences_found:
            print(f"\nAll {field_label} values match!")
    else:
        # Light output mode - just list files to download/fix
        all_needed_files = files_to_download + [f for f, _, _ in files_with_mismatch]
        for f in all_needed_files:
            print(f)

    return differences_found


def download_remote_csv(ssh_host, remote_path, local_csv_file):
    """Download CSV file from remote host if it exists."""
    remote_csv = f"{remote_path}/sha256-list.csv"
    try:
        result = subprocess.run(
            ['ssh', ssh_host, f'test -f "{remote_csv}" && echo "exists"'],
            capture_output=True,
            text=True,
            check=False
        )
        if result.stdout.strip() == "exists":
            print(f"Downloading existing CSV from remote: {remote_csv}")
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
    """Upload CSV file to remote host."""
    remote_csv = f"{remote_path}/sha256-list.csv"
    try:
        print(f"Uploading CSV to remote: {remote_csv}")
        subprocess.run(
            ['scp', local_csv_file, f'{ssh_host}:{remote_csv}'],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"  Uploaded: {local_csv_file} -> {ssh_host}:{remote_csv}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Error uploading CSV to remote: {e}")
        return False


def initialize_csv_from_directory(directory, csv_file, dry_run=False, ssh_host=None, remote_path=None):
    """Create CSV file with filenames if it doesn't exist."""
    if not os.path.exists(csv_file):
        if ssh_host:
            tensordata_files = list_remote_files(ssh_host, remote_path)
        else:
            tensordata_files = sorted([
                f for f in os.listdir(directory)
                if (f.endswith('.ckpt-tensordata') or f.endswith('.ckpt')) and os.path.isfile(os.path.join(directory, f))
            ])

        if dry_run:
            print(f"Would create CSV file: {csv_file}")
            print(f"Would add {len(tensordata_files)} file(s) to CSV")
            return

        print(f"Creating new CSV file: {csv_file}")
        checksums = {}
        for filename in tensordata_files:
            checksums[filename] = {'sha256sum': '', '8k_sha256sum': '', 'filesize': ''}
        write_csv(csv_file, checksums)
        print(f"Created CSV with {len(tensordata_files)} files")


def main():
    # Check for flags
    dry_run = '--dry-run' in sys.argv
    verbose = '--verbose' in sys.argv or '-v' in sys.argv
    force = '--force' in sys.argv

    # Parse level flag
    level = 'L3'  # Default
    for arg in sys.argv:
        if arg.upper() in ['L1', 'L2', 'L3', 'ALL']:
            level = arg.upper()
            break
        if arg.startswith('--level='):
            level = arg.split('=')[1].upper()
            break

    # Parse --file argument (can be specified multiple times or comma-separated)
    specific_files = []
    args_to_filter = []
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == '--file' and i + 1 < len(sys.argv):
            # Handle --file filename format
            file_arg = sys.argv[i + 1]
            # Support comma-separated files
            specific_files.extend([f.strip() for f in file_arg.split(',') if f.strip()])
            args_to_filter.extend([arg, sys.argv[i + 1]])
            i += 2
        elif arg.startswith('--file='):
            # Handle --file=filename format
            file_arg = arg.split('=', 1)[1]
            # Support comma-separated files
            specific_files.extend([f.strip() for f in file_arg.split(',') if f.strip()])
            args_to_filter.append(arg)
            i += 1
        else:
            i += 1

    args = [arg for arg in sys.argv[1:] if arg not in ['--dry-run', '--verbose', '-v', '--force', 'L1', 'L2', 'L3', 'l1', 'l2', 'l3', 'all', 'ALL'] + args_to_filter and not arg.startswith('--level=') and not arg.startswith('--file=')]

    if len(args) < 1 or len(args) > 2:
        print("Usage: python compare_checksums.py [options] <directory|csv1> [csv2]")
        print()
        print("Options:")
        print("  L1, L2, L3, all or --level=L1/L2/L3/all")
        print("       L1:  File size comparison (fastest)")
        print("       L2:  8K hash (4KB from start + 4KB from end)")
        print("       L3:  Full SHA256 hash (default, slowest)")
        print("       all: Run L1, L2, L3 sequentially (fills missing values only)")
        print("  --file=<filename>: Update only specific file(s). Can be:")
        print("       - Single file: --file=model.ckpt")
        print("       - Comma-separated: --file=model1.ckpt,model2.ckpt")
        print("       - Multiple flags: --file=model1.ckpt --file=model2.ckpt")
        print("       Note: Only updates the specified level; other levels unchanged")
        print("  --dry-run: Print what would be done without writing files")
        print("  --verbose, -v: Show detailed output in comparison mode")
        print("  --force: Recalculate all values even if they exist")
        print()
        print("CSV format: filename, sha256sum, 8k_sha256sum, filesize")
        print()
        print("Modes:")
        print("  1. Single directory mode: Update sha256-list.csv at specified level")
        print("     Local:  ./compare_checksums_v2.py L1 /path/to/dir1")
        print("     Remote: ./compare_checksums_v2.py L2 root@host:/path/to/dir1")
        print()
        print("  2. CSV comparison mode: Compare two CSV files at specified level")
        print("     Example: ./compare_checksums_v2.py L1 csv1.csv csv2.csv")
        print("     Output: List of files with differences (one per line)")
        print("     Note: csv2 is treated as the source of truth")
        print()
        print("Examples:")
        print("  # Update file sizes (L1) for local directory")
        print("  ./compare_checksums.py L1 /mnt/models/official-models")
        print()
        print("  # Update 8K hashes (L2) for remote directory")
        print("  ./compare_checksums.py L2 root@server:/mnt/models")
        print()
        print("  # Update all levels (L1 + L2 + L3) for directory")
        print("  ./compare_checksums.py all /mnt/models/official-models")
        print()
        print("  # Update only L2 for a specific file (L1 and L3 unchanged)")
        print("  ./compare_checksums.py L2 --file=model.ckpt /mnt/models")
        print()
        print("  # Update L1 for multiple specific files")
        print("  ./compare_checksums.py L1 --file=model1.ckpt,model2.ckpt /mnt/models")
        print()
        print("  # Update L3 for specific files on remote server")
        print("  ./compare_checksums.py L3 --file=model.ckpt root@server:/mnt/models")
        print()
        print("  # Compare two CSVs using file size (L1)")
        print("  ./compare_checksums.py L1 local.csv remote.csv")
        print()
        print("  # Compare two CSVs at all levels")
        print("  ./compare_checksums.py all local.csv remote.csv")
        sys.exit(1)

    first_arg = args[0]

    # Check if second argument is provided
    if len(args) == 2:
        second_arg = args[1]

        # CSV comparison mode - both arguments should be CSV files
        if first_arg.endswith('.csv') and second_arg.endswith('.csv'):
            if force:
                print("Warning: --force flag is ignored in CSV comparison mode")
            if specific_files:
                print("Warning: --file flag is ignored in CSV comparison mode")

            if not os.path.isfile(first_arg):
                print(f"Error: {first_arg} is not a valid file")
                sys.exit(1)
            if not os.path.isfile(second_arg):
                print(f"Error: {second_arg} is not a valid file")
                sys.exit(1)

            if verbose:
                print(f"CSV comparison mode at level {level}")

            has_differences = compare_checksums(first_arg, second_arg, level, verbose)
            sys.exit(1 if has_differences else 0)
        else:
            print("Error: For CSV comparison mode, both arguments must be CSV files")
            sys.exit(1)
    else:
        # Single directory mode (local or remote)
        dir_arg = first_arg

        # Parse the directory argument (could be remote)
        is_remote, ssh_host, dir_path = parse_remote_path(dir_arg)

        # Determine which levels to process
        if level == 'ALL':
            levels_to_process = ['L1', 'L2', 'L3']
        else:
            levels_to_process = [level]

        if is_remote:
            print(f"Remote directory mode: {ssh_host}:{dir_path}")
            print(f"Level: {level}")

            # Create a local CSV file based on the remote hostname
            script_dir = Path(__file__).parent
            logs_dir = script_dir / "logs"
            logs_dir.mkdir(exist_ok=True)

            hostname = ssh_host.split('@')[-1] if '@' in ssh_host else ssh_host
            csv_file1 = str(logs_dir / f"sha256-list-{hostname}.csv")

            # Always download remote CSV to get accurate state (even in dry-run)
            download_remote_csv(ssh_host, dir_path, csv_file1)

            initialize_csv_from_directory(None, csv_file1, dry_run, ssh_host, dir_path)

            for lvl in levels_to_process:
                update_csv_with_level(None, csv_file1, lvl, dry_run, ssh_host, dir_path, force, specific_files if specific_files else None)

            if not dry_run:
                upload_csv_to_remote(ssh_host, dir_path, csv_file1)
        else:
            dir1 = dir_path

            if not os.path.isdir(dir1):
                print(f"Error: {dir1} is not a valid directory")
                sys.exit(1)

            csv_file1 = os.path.join(dir1, 'sha256-list.csv')

            initialize_csv_from_directory(dir1, csv_file1, dry_run)

            for lvl in levels_to_process:
                update_csv_with_level(dir1, csv_file1, lvl, dry_run, force=force, specific_files=specific_files if specific_files else None)

        if dry_run:
            print("\nDry-run completed (no files modified)")
        else:
            print(f"\nLevel {level} processing completed successfully")
        sys.exit(0)


if __name__ == '__main__':
    main()
