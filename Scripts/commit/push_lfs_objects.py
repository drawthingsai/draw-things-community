import os
import re
import shutil
import subprocess
import sys

# Run in the destination repository after apply.py commits. A synced patch only carries
# Git LFS pointers, so upload the objects they point to from the source repository before
# `git push`, otherwise the destination repository references objects it does not have.

POINTER_VERSION = b'version https://git-lfs.github.com/spec/v1\n'
POINTER_OID_PATTERN = re.compile(rb'^oid sha256:([0-9a-f]{64})$', re.MULTILINE)

def lfs_objects_dir(repository):
    git_dir = subprocess.check_output(['git', 'rev-parse', '--path-format=absolute', '--git-common-dir'], cwd=repository).decode('utf-8').strip()
    return os.path.join(git_dir, 'lfs', 'objects')

def object_path(objects_dir, oid):
    return os.path.join(objects_dir, oid[0:2], oid[2:4], oid)

def push_lfs_objects(source_repository):
    # Blobs added or changed by the commits that are not on the remote yet.
    cmd = ['git', 'diff', '--raw', '-z', '--no-renames', '--no-abbrev', '--diff-filter=AMT', '@{upstream}', 'HEAD']
    fields = subprocess.check_output(cmd).decode('utf-8').split('\0')
    pointers = {}
    for status, file in zip(fields[0::2], fields[1::2]):
        _, mode, _, blob, _ = status.split()
        # Pointers are small regular files; skip submodules and anything larger.
        if mode == '160000' or int(subprocess.check_output(['git', 'cat-file', '-s', blob]).decode('utf-8')) >= 1024:
            continue
        content = subprocess.check_output(['git', 'cat-file', 'blob', blob])
        match = POINTER_OID_PATTERN.search(content)
        if content.startswith(POINTER_VERSION) and match:
            pointers[match.group(1).decode('utf-8')] = (file, content)
    if not pointers:
        print('No Git LFS objects to push.')
        return
    source_objects_dir = lfs_objects_dir(source_repository)
    destination_objects_dir = lfs_objects_dir('.')
    for oid, (file, content) in pointers.items():
        source_object = object_path(source_objects_dir, oid)
        if not os.path.exists(source_object):
            # Smudging the pointer downloads the object into the source repository's storage.
            subprocess.run(['git', 'lfs', 'smudge', '--', file], cwd=source_repository, input=content, stdout=subprocess.DEVNULL, check=True)
        destination_object = object_path(destination_objects_dir, oid)
        os.makedirs(os.path.dirname(destination_object), exist_ok=True)
        shutil.copyfile(source_object, destination_object)
        print(f'Git LFS object {oid} for {file}')
    subprocess.run(['git', 'lfs', 'push', '--object-id', 'origin'] + list(pointers.keys()), check=True)

if len(sys.argv) != 2:
    print(f'Usage: {sys.argv[0]} <source repository>')
    sys.exit(1)
push_lfs_objects(sys.argv[1])
