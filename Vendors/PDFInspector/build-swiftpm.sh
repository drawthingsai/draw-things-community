#!/bin/bash
# Build the native archive used by the diagnostic SwiftPM Catalyst application.
set -euo pipefail
vendor_dir="$(cd "$(dirname "$0")" && pwd)"
repo_dir="$(cd "$vendor_dir/../.." && pwd)"
build_dir="$repo_dir/.build/pdf-inspector"
target="${1:-aarch64-apple-ios-macabi}"
revision=2dbd16b3ea4ea943af2802bd72e45dd264eacb9f
checksum=814d801ce6032f47a68e758c643b983b07e5949c674249e2a0ed463c6246264b
mkdir -p "$build_dir/source" "$build_dir/lib"
archive="$build_dir/$revision.tar.gz"
if [[ ! -f "$archive" ]]; then
  curl --fail --location "https://github.com/firecrawl/pdf-inspector/archive/$revision.tar.gz" -o "$archive"
fi
printf '%s  %s\n' "$checksum" "$archive" | shasum -a 256 --check
# Refresh only the source files, retaining Cargo's compilation cache.
tar -xzf "$archive" --strip-components=1 -C "$build_dir/source"
patch -d "$build_dir/source" -p1 < "$vendor_dir/password.patch"
cp "$vendor_dir/bridge.rs" "$build_dir/bridge.rs"
python3 - "$vendor_dir" "$build_dir" <<'PY'
from pathlib import Path
import sys
vendor, build = map(Path, sys.argv[1:])
# Use the same native-only features and dependency versions as the Bazel build.
manifest = (vendor / 'Cargo.toml').read_text()
manifest = manifest.replace('name = "pdf-inspector-dependencies"', 'name = "pdf-inspector"')
manifest = manifest.replace('version = "0.0.0"', 'version = "1.25.0"', 1)
manifest = manifest.replace('path = "fake.rs"', 'path = "src/lib.rs"')
manifest += '\n[features]\ndefault = []\n'
(build / 'source/Cargo.toml').write_text(manifest)
(build / 'Cargo.toml').write_text('''[package]
name = "pdf-inspector-bridge"
version = "0.0.0"
edition = "2021"
[lib]
path = "bridge.rs"
crate-type = ["staticlib"]
[dependencies]
pdf-inspector = { path = "source" }
''')
lock = (vendor / 'Cargo.lock').read_text().replace(
    'name = "pdf-inspector-dependencies"\nversion = "0.0.0"',
    'name = "pdf-inspector"\nversion = "1.25.0"')
lock += '\n[[package]]\nname = "pdf-inspector-bridge"\nversion = "0.0.0"\ndependencies = ["pdf-inspector"]\n'
(build / 'Cargo.lock').write_text(lock)
PY
cargo build --locked --release --target "$target" --manifest-path "$build_dir/Cargo.toml"
cp "$build_dir/target/$target/release/libpdf_inspector_bridge.a" "$build_dir/lib/"
printf 'Built %s for %s\n' "$build_dir/lib/libpdf_inspector_bridge.a" "$target"
