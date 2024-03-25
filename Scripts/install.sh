#!/usr/bin/env bash

set -euo pipefail

GIT_CONFIG=$(git rev-parse --git-dir)
GIT_ROOT=$(git rev-parse --show-toplevel)

mkdir -p $GIT_CONFIG/hooks/pre-commit.d

rm -f $GIT_CONFIG/hooks/pre-commit
ln -s $GIT_ROOT/Scripts/vendors/dispatch $GIT_CONFIG/hooks/pre-commit

rm -f $GIT_CONFIG/hooks/pre-commit.d/swift-format
ln -s $GIT_ROOT/Scripts/swift-format/pre-commit $GIT_CONFIG/hooks/pre-commit.d/swift-format

rm -f $GIT_CONFIG/hooks/pre-commit.d/buildifier
ln -s $GIT_ROOT/Scripts/buildifier/pre-commit $GIT_CONFIG/hooks/pre-commit.d/buildifier

OS=$(uname -s)

if [ "$OS" == "Darwin" ]; then
  echo "try-import %workspace%/.bazelrc.darwin" > $GIT_ROOT/.bazelrc
  ln -s $GIT_ROOT/WORKSPACE.darwin $GIT_ROOT/WORKSPACE
  # Install realpath
  brew install coreutils
  # Install xcode cmd line tools
  xcode-select --install
else
  echo "try-import %workspace%/.bazelrc.linux" > $GIT_ROOT/.bazelrc
  ln -s $GIT_ROOT/WORKSPACE.linux $GIT_ROOT/WORKSPACE
  $GIT_ROOT/Scripts/setup_clang.sh /usr/local
fi
