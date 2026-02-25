#!/bin/bash
# Generate DataModels Swift files from .fbs schemas using Bazel

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
OUTPUT_DIR="$PROJECT_ROOT/Libraries/DataModels/PreGeneratedSPM"

cd "$PROJECT_ROOT"

# Build all schema targets
echo "Building DataModels schemas with Bazel..."
bazel build \
    //Libraries/DataModels:config_schema \
    //Libraries/DataModels:estimation_schema \
    //Libraries/DataModels:mixing_schema \
    //Libraries/DataModels:lora_trainer_schema \
    //Libraries/DataModels:dataset_schema \
    //Libraries/DataModels:paint_color_schema \
    //Libraries/DataModels:peer_connection_id_schema

# Clear and copy generated files
echo "Copying generated files to $OUTPUT_DIR..."
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Copy all generated Swift files (skip JSON files)
find bazel-bin/Libraries/DataModels -name "*_generated.swift" -exec cp {} "$OUTPUT_DIR/" \;

echo "Formatting generated Swift files with swift-format..."
swift_files=()
while IFS= read -r -d '' file; do
    swift_files+=("$file")
done < <(find "$OUTPUT_DIR" -name "*.swift" -print0)

if [[ ${#swift_files[@]} -gt 0 ]]; then
    if [[ "$(uname)" == "Darwin" ]]; then
        export DYLD_FALLBACK_LIBRARY_PATH="$(xcode-select -p)/Toolchains/XcodeDefault.xctoolchain/usr/lib/swift/macosx"
    fi
    bazel run --compilation_mode=opt @SwiftFormat//:swift-format -- \
        format --configuration "$PROJECT_ROOT/.swift-format.json" -i "${swift_files[@]}"
fi

echo "Generated files:"
ls -la "$OUTPUT_DIR/"

echo "Done!"
