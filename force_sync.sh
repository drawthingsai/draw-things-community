#!/bin/bash

# Replaces the paths listed in the source repository's SYNCLIST for this repository with
# their copies from the source repository. Only files tracked there are copied, so untracked
# files in the source working tree never land here. A SYNCLIST line is either a path with
# the same name in both repositories, or `source path -> path here`.

cd "$(dirname "$0")" || exit 1

# Source repository path
SOURCE_REPO="../draw-things"

# File in the source repository containing the list of paths
DIR_LIST_FILE="SYNCLIST.draw-things-community"

# Destination path where paths will be copied
DESTINATION_PATH="./"

# Check if the path list file exists
if [[ ! -f "$SOURCE_REPO/$DIR_LIST_FILE" ]]; then
    echo "Path list file '$SOURCE_REPO/$DIR_LIST_FILE' not found!"
    exit 1
fi

STAGING_PATH="$(mktemp -d)"
trap 'rm -rf "$STAGING_PATH"' EXIT

# Read paths from the file and process them
while IFS= read -r line || [[ -n "$line" ]]; do
    # Skip empty lines
    if [[ -z "${line//[[:space:]]/}" ]]; then
        continue
    fi

    if [[ "$line" == *"->"* ]]; then
        source="${line%%->*}"
        destination="${line#*->}"
    else
        source="$line"
        destination="$line"
    fi
    source="$(sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//; s#/+$##' <<< "$source")"
    destination="$(sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//; s#/+$##' <<< "$destination")"
    if [[ -z "$source" || -z "$destination" ]]; then
        echo "Invalid line: $line"
        exit 1
    fi

    # Remove the path if it exists
    rm -rf "${DESTINATION_PATH:?}/$destination"
    echo "Removed: $DESTINATION_PATH$destination"

    # Copy the tracked files from the source repo
    git -C "$SOURCE_REPO" --literal-pathspecs ls-files -z -- "$source" \
        | (cd "$SOURCE_REPO" && COPYFILE_DISABLE=1 tar --null -T - -cf -) \
        | tar -C "$STAGING_PATH" -xf -
    if [[ -e "$STAGING_PATH/$source" || -L "$STAGING_PATH/$source" ]]; then
        mkdir -p "$(dirname "$DESTINATION_PATH$destination")"
        mv "$STAGING_PATH/$source" "$DESTINATION_PATH$destination"
        echo "Copied: $source from $SOURCE_REPO to $DESTINATION_PATH$destination"
    else
        echo "No tracked files at $source in $SOURCE_REPO"
    fi

done < "$SOURCE_REPO/$DIR_LIST_FILE"

echo "Process completed."
