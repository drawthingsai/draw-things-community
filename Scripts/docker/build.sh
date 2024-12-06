#!/usr/bin/env bash

set -euo pipefail

docker build -f cudaSwiftDockerfile -t cuda-swift:cuda12.4.1-cudnn-runtime-ubuntu22.04-swift5.10.1 .
