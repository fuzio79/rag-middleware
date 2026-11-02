#!/usr/bin/env bash
# Drop the LanceDB `documents` table. DESTRUCTIVE: every ingested chunk is lost.
#
# Requires an explicit confirmation flag. Refuses to run by accident.
#
# This runs on the HOST, against the same database directory the container
# mounts, so the service does not need to be running. Point RAG_DB_PATH at the
# host side of the knowledge-base volume (default ./data/knowledge_base, which
# is what docker-compose.yml mounts).
set -euo pipefail

DB_PATH="${RAG_DB_PATH:-./data/knowledge_base}"

if [ "${1:-}" != "--yes-i-know" ]; then
    cat >&2 <<EOF
This deletes the 'documents' table in: ${DB_PATH}
Every ingested document and web page will be lost.

Re-run with the confirmation flag if that is what you want:
    $0 --yes-i-know
EOF
    exit 2
fi

echo "[*] Dropping table 'documents' from ${DB_PATH}"
python3 - "${DB_PATH}" <<'PY'
import sys
import lancedb

db = lancedb.connect(sys.argv[1])
try:
    db.drop_table("documents")
    print("[ok] table dropped")
except Exception as exc:  # table may simply not exist
    print(f"[!] could not drop table: {exc}")
finally:
    db.close()
PY
