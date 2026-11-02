#!/usr/bin/env bash
# Upload every supported document under a directory into the knowledge base.
#
#   RAG_API_KEY=... ./scripts/upload.sh <directory> [rag-base-url]
#
# The server authenticates the upload endpoint, so RAG_API_KEY is required
# unless the server runs with RAG_REQUIRE_AUTH=false.
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <directory_path> [rag_base_url]" >&2
    exit 2
fi

TARGET_DIR="$1"
BASE_URL="${2:-${RAG_HOST:-http://localhost:8000}}"
UPLOAD_URL="${BASE_URL%/}/api/v1/knowledge/upload_files/"
API_KEY="${RAG_API_KEY:-}"

AUTH=()
if [ -n "$API_KEY" ]; then
    AUTH=(-H "X-API-Key: ${API_KEY}")
else
    echo "[!] RAG_API_KEY is empty; uploads will be rejected with 401 unless" >&2
    echo "[!] the server runs with RAG_REQUIRE_AUTH=false." >&2
fi

if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: directory '$TARGET_DIR' not found." >&2
    exit 1
fi

echo "[*] Uploading from: $TARGET_DIR"
echo "[*] Endpoint     : $UPLOAD_URL"

failed=0
while IFS= read -r -d '' file_path; do
    printf ' -> %s ... ' "$file_path"
    # --fail turns HTTP 4xx/5xx into a non-zero exit, so a server-side
    # rejection can no longer be reported as [SUCCESS].
    if curl -fsS -X POST "$UPLOAD_URL" \
            "${AUTH[@]}" \
            -H 'accept: application/json' \
            -F "files=@${file_path}" > /dev/null; then
        echo 'ok'
    else
        echo 'FAILED'
        failed=$((failed + 1))
    fi
done < <(find "$TARGET_DIR" -type f \
    \( -iname '*.pdf' -o -iname '*.txt' -o -iname '*.docx' -o -iname '*.eml' \) -print0)

echo
echo "[*] Finished. failures: ${failed}"
[ "$failed" -eq 0 ]
