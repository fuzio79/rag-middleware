#!/usr/bin/env bash
# Smoke tests for the ingest + search endpoints.
#
#   RAG_HOST=http://127.0.0.1:8000 RAG_API_KEY=... ./scripts/test-scrap.sh
#
# The server authenticates every data endpoint, so RAG_API_KEY is required
# unless the server runs with RAG_REQUIRE_AUTH=false.
set -euo pipefail

RAG_HOST="${RAG_HOST:-http://localhost:8000}"
API_KEY="${RAG_API_KEY:-}"

# Build the auth header once. An empty key is omitted rather than sent blank,
# so the failure the server reports is "missing", not "wrong".
AUTH=()
if [ -n "$API_KEY" ]; then
    AUTH=(-H "X-API-Key: ${API_KEY}")
else
    echo "[!] RAG_API_KEY is empty; every data call below will be rejected"
    echo "[!] with 401 unless the server runs with RAG_REQUIRE_AUTH=false."
fi

echo "[*] Target: ${RAG_HOST}"

echo "[*] 1. Liveness"
curl -fsS "${RAG_HOST}/health"
echo

echo "[*] 2. Registered ingest routes"
curl -fsS "${RAG_HOST}/openapi.json" | python3 -c "
import sys, json
paths = json.load(sys.stdin)['paths']
print([p for p in paths if 'ingest' in p])
"

echo "[*] 3. Ingest smoke test"
curl -fsS -X POST "${RAG_HOST}/api/v1/rag/ingest" \
  "${AUTH[@]}" \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://en.wikipedia.org/wiki/Vector_database","engine":"auto"}'
echo

echo "[*] 4. Retrieval smoke test"
curl -fsS -X POST "${RAG_HOST}/api/v1/rag/search" \
  "${AUTH[@]}" \
  -H 'Content-Type: application/json' \
  -d '{"query":"what is a vector database"}'
echo

echo "[ok] done"
