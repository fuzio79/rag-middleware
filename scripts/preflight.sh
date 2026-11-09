#!/usr/bin/env bash
# Pre-flight checks for a rag-middleware deployment.
#
# Every check here maps to a finding in the code review that no in-process code
# change can close, or to an invariant the code now relies on. It reports and
# exits non-zero rather than fixing anything: these are deployment decisions.
#
# Run from the repository root on the host, or inside the rag-service container
# with --in-container (which needs only python + the service's own env).
set -euo pipefail

IN_CONTAINER=0
CRAWLER_CONTAINER="${CRAWLER_CONTAINER:-crawl4ai}"
SERVICE_CONTAINER="${SERVICE_CONTAINER:-rag-service}"

for arg in "$@"; do
  case "$arg" in
    --in-container) IN_CONTAINER=1 ;;
    -h|--help)
      sed -n '2,12p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *)
      echo "unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

FAILURES=0
WARNINGS=0

pass() { printf '  [ OK ] %s\n' "$1"; }
warn() { printf '  [WARN] %s\n' "$1"; WARNINGS=$((WARNINGS + 1)); }
fail() { printf '  [FAIL] %s\n' "$1"; FAILURES=$((FAILURES + 1)); }
head2() { printf '\n%s\n' "$1"; }

# `case` rather than `printf | grep -q`: under `set -o pipefail` an early-exiting
# grep gives printf SIGPIPE (141) and the pipeline reports failure, so a match
# can be read as "not found".

# --- 1. single worker ------------------------------------------------------
# store._lock_for is an in-process lock. With more than one uvicorn worker the
# check-then-write sequence is atomic per process only, so the duplicate-row
# protection this branch adds stops holding -- silently.
head2 "1. uvicorn worker count"
CMDLINE=""
if [ "$IN_CONTAINER" -eq 1 ]; then
  CMDLINE="$(tr '\0' ' ' </proc/1/cmdline 2>/dev/null || true)"
elif command -v docker >/dev/null 2>&1; then
  CMDLINE="$(docker exec "$SERVICE_CONTAINER" sh -c 'tr "\0" " " </proc/1/cmdline' 2>/dev/null || true)"
fi

case "$CMDLINE" in
  *--workers*)
    WORKERS="$(printf '%s' "$CMDLINE" | sed -n 's/.*--workers[= ]*\([0-9][0-9]*\).*/\1/p')"
    if [ -n "$WORKERS" ] && [ "$WORKERS" -gt 1 ]; then
      fail "running with --workers ${WORKERS}; the per-source lock is in-process only"
    else
      pass "single worker"
    fi
    ;;
  "")
    warn "could not read the service process cmdline; check the worker count by hand"
    ;;
  *)
    pass "no --workers flag (uvicorn defaults to 1)"
    ;;
esac

# --- 2. crawler egress -----------------------------------------------------
# The real SSRF control. The in-process guard cannot see JS redirects, meta
# refresh, or DNS rebinding, so what actually bounds the crawler is where it
# can reach. If it can open a socket to another service on this network, a
# redirect chain is enough to pull that service's content into the KB.
head2 "2. crawler egress containment"
if [ "$IN_CONTAINER" -eq 1 ]; then
  warn "cannot test crawler egress from inside rag-service; run on the host"
elif ! command -v docker >/dev/null 2>&1; then
  # Testing CMDLINE here was wrong: a failed cmdline read is not "no docker",
  # and with docker absent the else-branch below would run `docker exec`, get
  # 000 for every target, and report a confident PASS on a security check that
  # never ran.
  warn "docker not available; crawler egress NOT verified"
else
  REACHED=0
  for target in "rag-service:8000" "open-webui:8080"; do
    host="${target%%:*}"
    port="${target##*:}"
    # Exit status, not the HTTP code. The previous version probed with
    # `curl -w '%{http_code}' A || curl -w ... B || echo 000` and treated any
    # non-empty, non-"000" result as reachable. curl emits "000" and exits
    # non-zero when it cannot connect, so an unreachable target produced
    # "000000000" from the three fallbacks concatenated -- non-empty and not
    # equal to "000", i.e. a confident FALSE POSITIVE. A correctly firewalled
    # crawler was reported as able to reach its siblings.
    #
    # Symptom it also hid: an HTTP 404 from a reachable host was not "/health",
    # so the second fallback fired and the reported code was not the first's.
    if docker exec "$CRAWLER_CONTAINER" sh -c \
        "curl -s -o /dev/null --connect-timeout 3 -m 5 'http://${target}/'" \
        >/dev/null 2>&1; then
      warn "crawler can reach ${target} -- add egress filtering"
      REACHED=$((REACHED + 1))
    fi
  done

  # The bridge gateway is where the Docker API and any host-published services
  # live, so it is the highest-value target and was not probed at all.
  GW="$(docker network inspect bridge \
        --format '{{range .IPAM.Config}}{{.Gateway}}{{end}}' 2>/dev/null || true)"
  if [ -n "$GW" ]; then
    for port in 8091 2375 2376; do
      if docker exec "$CRAWLER_CONTAINER" sh -c \
          "curl -s -o /dev/null --connect-timeout 3 -m 5 'http://${GW}:${port}/'" \
          >/dev/null 2>&1; then
        warn "crawler can reach the docker bridge gateway on :${port} (${GW})"
        REACHED=$((REACHED + 1))
      fi
    done
  fi

  if [ "$REACHED" -eq 0 ]; then
    pass "crawler could not reach sibling services or the gateway"
  fi
fi

# --- 3. crawl allowlist ----------------------------------------------------
# Deliberate policy for this deployment: anything that is NOT local is allowed.
# Empty RAG_ALLOWED_CRAWL_DOMAINS is exactly that -- every public host is
# reachable, and validate_crawl_url still rejects loopback, private,
# link-local and reserved ranges. So an empty value is the intended
# configuration, not a misconfiguration, and is reported as OK.
head2 "3. RAG_ALLOWED_CRAWL_DOMAINS"
ALLOWED="${RAG_ALLOWED_CRAWL_DOMAINS:-}"
if [ "$IN_CONTAINER" -eq 0 ] && [ -z "$ALLOWED" ] && command -v docker >/dev/null 2>&1; then
  ALLOWED="$(docker exec "$SERVICE_CONTAINER" printenv RAG_ALLOWED_CRAWL_DOMAINS 2>/dev/null || true)"
fi
if [ -z "$ALLOWED" ]; then
  pass "public-only policy: any non-local host is crawlable, private ranges rejected"
else
  pass "narrowed allowlist: ${ALLOWED}"
fi

# --- 4. auth ---------------------------------------------------------------
head2 "4. API key and auth mode"
if [ "$IN_CONTAINER" -eq 1 ]; then
  KEY="${RAG_API_KEY:-}"
  AUTH_REQ="${RAG_REQUIRE_AUTH:-true}"
else
  KEY="$(docker exec "$SERVICE_CONTAINER" printenv RAG_API_KEY 2>/dev/null || true)"
  AUTH_REQ="$(docker exec "$SERVICE_CONTAINER" printenv RAG_REQUIRE_AUTH 2>/dev/null || echo true)"
fi
case "${AUTH_REQ,,}" in
  false|0|no|off)
    warn "RAG_REQUIRE_AUTH=${AUTH_REQ}: every endpoint is open (trusted network only)"
    ;;
  *)
    if [ -n "$KEY" ]; then
      pass "auth required and a key is set"
    else
      fail "auth required but RAG_API_KEY is empty: data endpoints answer 503"
    fi
    ;;
esac

# --- 5. docs gate ----------------------------------------------------------
head2 "5. /docs and /openapi.json"
PUBLIC_DOCS="${RAG_PUBLIC_DOCS:-false}"
if [ "$IN_CONTAINER" -eq 0 ] && command -v docker >/dev/null 2>&1; then
  PUBLIC_DOCS="$(docker exec "$SERVICE_CONTAINER" printenv RAG_PUBLIC_DOCS 2>/dev/null || echo false)"
fi
case "${PUBLIC_DOCS,,}" in
  true|1|yes|on)
    warn "RAG_PUBLIC_DOCS=${PUBLIC_DOCS}: the schema is served without a key"
    ;;
  *)
    pass "docs gated (RAG_PUBLIC_DOCS not enabled)"
    ;;
esac

# --- 6. vector table health ------------------------------------------------
# Two things worth catching on an existing database:
#   * a table built by a different embedding model (silently incomparable vectors)
#   * duplicate generations per source, left by the add-then-delete ordering
#     that this branch replaces. Newer ingests repair a source as they touch it,
#     so an old duplicate only disappears when that source is re-ingested.
head2 "6. vector table"
if [ "$IN_CONTAINER" -eq 0 ] && command -v docker >/dev/null 2>&1; then
  PYTHON_CMD=(docker exec "$SERVICE_CONTAINER" python)
else
  PYTHON="python3"
  command -v python3 >/dev/null 2>&1 || PYTHON="python"
  PYTHON_CMD=("$PYTHON")
fi

TABLE_REPORT="$("${PYTHON_CMD[@]}" - <<'PY' 2>/dev/null || true
import os
try:
    import lancedb
except Exception:
    print("SKIP no-lancedb")
    raise SystemExit(0)
path = os.getenv("RAG_DB_PATH", "local_knowledge_base")
try:
    db = lancedb.connect(path)
    if "documents" not in list(db.table_names()):
        print("SKIP no-table")
        raise SystemExit(0)
    t = db.open_table("documents")
    total = t.count_rows()
    rows = t.search().select(["source", "file_hash"]).limit(100000).to_list()
    generations = {}
    for r in rows:
        generations.setdefault(r["source"], set()).add(r["file_hash"])
    dupes = {s: len(h) for s, h in generations.items() if len(h) > 1}
    dim = t.schema.field("vector").type.list_size
    print(f"OK total={total} sources={len(generations)} dim={dim} dupes={len(dupes)}")
    if dupes:
        print("DUPES " + ", ".join(sorted(dupes)[:5]))
except Exception as exc:
    print(f"SKIP error:{type(exc).__name__}")
PY
)"

case "$TABLE_REPORT" in
  *no-lancedb*)
    warn "lancedb not importable; skipped table checks"
    ;;
  *no-table*)
    pass "no documents table yet (nothing indexed)"
    ;;
  *error:*)
    warn "could not read the table (${TABLE_REPORT#*error:})"
    ;;
  *OK*)
    pass "table readable: ${TABLE_REPORT#*OK }"
    case "$TABLE_REPORT" in
      *DUPES*)
        warn "sources with more than one stored generation: ${TABLE_REPORT#*DUPES }"
        warn "re-ingest each to drop the stale generation, or rebuild the table"
        ;;
    esac
    ;;
  "")
    warn "table probe produced no output"
    ;;
  *)
    warn "unexpected table report: ${TABLE_REPORT}"
    ;;
esac

# --- summary ---------------------------------------------------------------
printf '\n%s\n' "----------------------------------------"
printf 'failures: %d   warnings: %d\n' "$FAILURES" "$WARNINGS"
if [ "$FAILURES" -gt 0 ]; then
  echo "preflight FAILED: fix the [FAIL] items before relying on this deployment"
  exit 1
fi
if [ "$WARNINGS" -gt 0 ]; then
  echo "preflight passed with warnings: each one is a known deployment trade-off"
  exit 0
fi
echo "preflight clean"
