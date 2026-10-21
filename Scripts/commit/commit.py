import argparse
import subprocess
import os

# Each SYNCLIST line is either a path that keeps the same name in both repositories, or
# `source -> destination` when the path is named differently in the destination repository.
# Paths match whole path components, so `SYNCLIST` does not match `SYNCLIST.local-code-community`.

def read_mappings_from_file(file_path, reverse):
    try:
        with open(file_path, 'r') as file:
            lines = [line.strip() for line in file.readlines() if line.strip()]
    except FileNotFoundError:
        print(f'File not found: {file_path}')
        return []
    mappings = []
    for line in lines:
        source, arrow, destination = line.partition('->')
        source = source.strip().rstrip('/')
        destination = destination.strip().rstrip('/') if arrow else source
        mappings.append((destination, source) if reverse else (source, destination))
    # Prefer the most specific mapping when entries nest.
    mappings.sort(key=lambda mapping: len(mapping[0]), reverse=True)
    return mappings

def map_path(file, mappings):
    for source, destination in mappings:
        if file == source:
            return destination
        if file.startswith(source + '/'):
            return destination + file[len(source):]
    return None

def retarget_diff(diff_output, file, mapped_file):
    if file == mapped_file:
        return diff_output
    lines = diff_output.split('\n')
    if lines[0] != f'diff --git a/{file} b/{file}':
        raise ValueError(f'Cannot retarget diff header: {lines[0]}')
    lines[0] = f'diff --git a/{mapped_file} b/{mapped_file}'
    for i, line in enumerate(lines):
        # Only the header names the path; stop before the content of the diff.
        if line.startswith('@@') or line == 'GIT binary patch':
            break
        if line == f'--- a/{file}':
            lines[i] = f'--- a/{mapped_file}'
        elif line == f'+++ b/{file}':
            lines[i] = f'+++ b/{mapped_file}'
    return '\n'.join(lines)

def generate_custom_patch(mappings):
    try:
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()

        # Getting commit metadata
        author_name = subprocess.check_output(['git', 'log', '--format=%an', '-n', '1', commit_hash]).decode('utf-8').strip()
        author_email = subprocess.check_output(['git', 'log', '--format=%ae', '-n', '1', commit_hash]).decode('utf-8').strip()
        commit_date = subprocess.check_output(['git', 'log', '--format=%ad', '--date=iso', '-n', '1', commit_hash]).decode('utf-8').strip()
        commit_message = subprocess.check_output(['git', 'log', '--format=%B', '-n', '1', commit_hash]).decode('utf-8').strip()

        # Get list of modified files in the top commit that match the filter paths
        cmd = ['git', 'diff-tree', '--no-commit-id', '--name-only', '-z', '-r', commit_hash]
        modified_files = [file for file in subprocess.check_output(cmd).decode('utf-8').split('\0') if file]
        filtered_files = []
        for file in modified_files:
            mapped_file = map_path(file, mappings)
            if mapped_file is not None:
                filtered_files.append((file, mapped_file))

        # Generate diff for the filtered files and write to a patch file
        if filtered_files:
            patch_file_name = 'custom_patch_with_metadata.patch'
            with open(patch_file_name, 'w') as patch_file:
                patch_file.write(f'From: {author_name} <{author_email}>\n')
                patch_file.write(f'Date: {commit_date}\n')
                patch_file.write(f'Subject: [PATCH] {commit_message}\n')
                patch_content = f'\n\n---\n\n'
                for file, mapped_file in filtered_files:
                    diff_cmd = ['git', '-c', 'core.quotePath=false', 'diff', '--binary', commit_hash + '^', commit_hash, '--', file]
                    diff_output = subprocess.check_output(diff_cmd).decode('utf-8')
                    if diff_output:
                        patch_content += retarget_diff(diff_output, file, mapped_file)
                patch_file.write(patch_content)
            print(f'Patch file generated: {patch_file_name}')
        else:
            print('No modified files match the specified paths.')
    except subprocess.CalledProcessError as e:
        print(f'Error generating custom patch: {e}')

parser = argparse.ArgumentParser(description='Generate a patch of the HEAD commit for paths listed in a SYNCLIST.')
parser.add_argument('--synclist', default='SYNCLIST', help='SYNCLIST file, relative to the root directory of the repository.')
parser.add_argument('--reverse', action='store_true', help='Map `source -> destination` lines from destination back to source.')
args = parser.parse_args()

# Assuming the SYNCLIST is in the root directory of the repository
root_dir = subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).decode('utf-8').strip()
sync_list_path = os.path.join(root_dir, args.synclist)

# Read paths to filter from the SYNCLIST file
mappings = read_mappings_from_file(sync_list_path, args.reverse)

if mappings:
    generate_custom_patch(mappings)
else:
    print('No paths to filter. Please check the SYNCLIST file.')
