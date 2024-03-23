import subprocess
import re
import os

def apply_custom_patch(patch_file_path):
    # Read the patch file content
    with open(patch_file_path, 'r') as file:
        patch_content = file.read()
    
    # Extract metadata and diff
    commit_message_match = re.search(r'Subject: \[PATCH\] (.+)', patch_content)
    author_match = re.search(r'From: (.+)', patch_content)
    
    commit_message = commit_message_match.group(1) if commit_message_match else "Commit applied without original message"
    author = author_match.group(1) if author_match else ""

    # Isolate the diff part
    diff_content = patch_content.split('---\n\n', 1)[1] if '---\n\n' in patch_content else ""

    # Temporarily write the diff content to a file
    diff_file_path = 'temp_diff.patch'
    with open(diff_file_path, 'w') as diff_file:
        diff_file.write(diff_content)
    
    # Apply the diff using git apply
    try:
        subprocess.run(['git', 'apply', diff_file_path], check=True)
        print("Diffs applied successfully.")
    except subprocess.CalledProcessError as e:
        print("Failed to apply diffs:", e)
        return

    # Stage changes
    subprocess.run(['git', 'add', '.'], check=True)

    # Commit changes with extracted metadata
    commit_command = ['git', 'commit', '-m', commit_message]
    if author:
        commit_command += ['--author', author]
    try:
        subprocess.run(commit_command, check=True)
        print("Changes committed with extracted metadata.")
    except subprocess.CalledProcessError as e:
        print("Failed to commit changes:", e)

# Path to your custom patch file
patch_file_path = 'custom_patch_with_metadata.patch'
apply_custom_patch(patch_file_path)
