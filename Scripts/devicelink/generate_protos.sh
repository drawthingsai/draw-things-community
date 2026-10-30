#!/usr/bin/env bash
# Regenerates Libraries/DeviceLink/Sources/Generated from Libraries/DeviceLink/Protos with the
# protoc-gen-swift and protoc-gen-grpc-swift pinned in Bazel, so the generated code matches the
# SwiftProtobuf and gRPC runtimes. Version 2 (Protos/v2) also gets gRPC clients and providers.
set -euo pipefail

GIT_ROOT=$(git rev-parse --show-toplevel)
cd "$GIT_ROOT"

bazel build @SwiftProtobuf//:protoc-gen-swift @grpc-swift//:protoc-gen-grpc-swift
PLUGIN="$(bazel cquery @SwiftProtobuf//:protoc-gen-swift --output=files 2>/dev/null)"
GRPC_PLUGIN="$(bazel cquery @grpc-swift//:protoc-gen-grpc-swift --output=files 2>/dev/null)"

protoc \
  --plugin=protoc-gen-swift="$GIT_ROOT/$PLUGIN" \
  --swift_out=Libraries/DeviceLink/Sources/Generated \
  --swift_opt=Visibility=Public \
  --proto_path=Libraries/DeviceLink/Protos \
  Libraries/DeviceLink/Protos/*.proto

protoc \
  --plugin=protoc-gen-swift="$GIT_ROOT/$PLUGIN" \
  --plugin=protoc-gen-grpc-swift="$GIT_ROOT/$GRPC_PLUGIN" \
  --swift_out=Libraries/DeviceLink/Sources/Generated \
  --swift_opt=Visibility=Public \
  --grpc-swift_out=Libraries/DeviceLink/Sources/Generated \
  --grpc-swift_opt=Visibility=Public \
  --proto_path=Libraries/DeviceLink/Protos \
  Libraries/DeviceLink/Protos/v2/*.proto
