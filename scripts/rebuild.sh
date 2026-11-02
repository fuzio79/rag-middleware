#!/usr/bin/env bash
# Rebuild the rag-middleware image ONLY.
#
# This script must never stop or remove a container. Whatever stack runs the
# service owns its lifecycle; stopping it by name from here takes down
# production. Redeploy is out of scope for this repository.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-rag-middleware:latest}"

echo "[*] Building ${IMAGE} from ${REPO_ROOT}"
docker build -t "$IMAGE" "$REPO_ROOT"

echo
echo "[i] Image rebuilt. This script intentionally does not restart anything."
echo "[i] Redeploy through whatever compose project owns the running container."
echo "[i] Verify the image carries no VCS metadata or secrets:"
echo "      docker run --rm --entrypoint sh ${IMAGE} -c 'ls -a /usr/src/app; test ! -e /usr/src/app/.git'"
