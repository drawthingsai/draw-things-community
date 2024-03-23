import subprocess
import os

# Function to read paths from a file
def read_paths_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            return [line.strip() for line in file if line.strip()]
    except FileNotFoundError:
        print(f'File not found: {file_path}')
        return []

# Assuming SYNCLIST is in the root directory of the repository
root_dir = subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).decode('utf-8').strip()
sync_list_path = os.path.join(root_dir, 'SYNCLIST')

# Read paths to filter from SYNCLIST file
paths_to_filter = read_paths_from_file(sync_list_path)

# Proceed if paths were successfully read
if paths_to_filter:
    # Get the top commit's hash
    top_commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()

    # Get list of modified files in the top commit that match the filter paths
    cmd = ['git', 'diff-tree', '--no-commit-id', '--name-only', '-r', top_commit_hash]
    modified_files = subprocess.check_output(cmd).decode('utf-8').splitlines()
    filtered_files = [file for file in modified_files if any(file.startswith(path) for path in paths_to_filter)]

    # Generate diff for the filtered files and write to a patch file
    if filtered_files:
        top_commit_message = subprocess.check_output(['git', 'log', '--format=%B', '-n', '1', top_commit_hash]).decode('utf-8').strip()
        patch_file_name = 'custom_patch_with_metadata.patch'
        with open(patch_file_name, 'w') as patch_file:
            patch_content = f'From: {subprocess.check_output(["git", "log", "--format=%an", "-n", "1", top_commit_hash]).decode("utf-8").strip()}\n' \
                  f'Date: {subprocess.check_output(["git", "log", "--format=%ad", "-n", "1", top_commit_hash]).decode("utf-8").strip()}\n' \
                  f'Subject: [PATCH] {top_commit_message}\n' \
                  f'\n---\n\n'
            for file in filtered_files:
                diff_cmd = ['git', 'diff', top_commit_hash + '^', top_commit_hash, '--', file]
                diff_output = subprocess.check_output(diff_cmd).decode('utf-8')
                if diff_output:
                    patch_content += f'diff --git a/{file} b/{file}\n' + diff_output
            patch_file.write(patch_content)
        print(f'Patch file generated: {patch_file_name}')
    else:
        print('No modified files match the specified paths.')
else:
    print('No paths to filter. Please check the SYNCLIST file.')
