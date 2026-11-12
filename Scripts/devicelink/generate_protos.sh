#!/usr/bin/env bash
# Regenerates Device Access's room messages in Libraries/DeviceLink/Sources/Generated with the
# protoc-gen-swift pinned in Bazel, so the generated code matches the SwiftProtobuf runtime.
set -euo pipefail

GIT_ROOT=$(git rev-parse --show-toplevel)
cd "$GIT_ROOT"

bazel build @SwiftProtobuf//:protoc-gen-swift
PLUGIN="$(bazel cquery @SwiftProtobuf//:protoc-gen-swift --output=files 2>/dev/null)"

protoc \
  --plugin=protoc-gen-swift="$GIT_ROOT/$PLUGIN" \
  --swift_out=Libraries/DeviceLink/Sources/Generated \
  --swift_opt=Visibility=Public \
  --proto_path=Libraries/DeviceLink/Protos \
  Libraries/DeviceLink/Protos/*.proto
