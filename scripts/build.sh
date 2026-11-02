#!/usr/bin/env bash
# Build the image.
#
# The build context is the REPOSITORY ROOT, not the directory this script
# lives in: the Dockerfile COPYs main.py and rag/ from there. Resolving the
# root from BASH_SOURCE keeps this correct no matter what the caller's cwd
# is - a plain `cd "$(dirname "$0")"` would now point at scripts/ and the
# build would fail with "main.py not found".
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-rag-middleware:latest}"

echo "[*] Building ${IMAGE} from ${REPO_ROOT}"
docker build -t "$IMAGE" "$REPO_ROOT"
