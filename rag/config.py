"""Configuration, read once from the environment at import time.

Every tunable lives here so nothing is hidden inside an endpoint module. Tests
import this and monkeypatch individual attributes rather than re-reading env.
"""

from __future__ import annotations

import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag")


def _csv_set(value: str) -> frozenset[str]:
    return frozenset(x.strip().lower() for x in value.split(",") if x.strip())


# --- storage ---------------------------------------------------------------
DB_PATH = os.getenv("RAG_DB_PATH", "local_knowledge_base")
EMBEDDING_MODEL_NAME = os.getenv("RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# --- auth ------------------------------------------------------------------
# Shared API key. When unset the service still starts (so a misconfigured
# deployment is debuggable) but every protected endpoint refuses and a loud
# warning is logged at startup. Set RAG_REQUIRE_AUTH=false to opt out
# explicitly, which is only appropriate on a trusted network.
API_KEY = os.getenv("RAG_API_KEY", "")
REQUIRE_AUTH = os.getenv("RAG_REQUIRE_AUTH", "true").lower() not in {
    "0",
    "false",
    "no",
    "off",
}

# --- scraping --------------------------------------------------------------
CRAWL_URL = os.getenv("CRAWL_URL", "http://crawl4ai:11235/crawl")
CRAWL_TOKEN = os.getenv("CRAWL4AI_API_TOKEN")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")  # optional

MAX_SCRAPE_CHARS = int(os.getenv("RAG_MAX_SCRAPE_CHARS", "500000"))
MAX_LINKS_IN_RESPONSE = 50

# Hard ceiling on chunks produced by one request, across the main page and
# every followed link. Without this, max_follow x a large page is unbounded CPU
# embedding work from a single call.
MAX_CHUNKS_PER_REQUEST = int(os.getenv("RAG_MAX_CHUNKS_PER_REQUEST", "2000"))

# --- chunking --------------------------------------------------------------
# all-MiniLM-L6-v2 truncates at 256 tokens (~1000 chars); keep chunks below it.
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "100"))
EMBED_BATCH_SIZE = 64

# --- retrieval -------------------------------------------------------------
MAX_QUERY_LENGTH = 4000
TOP_K_RESULTS = 3
MAX_CONTEXT_CHARS = 4000
MAX_CHUNK_CHARS = 1200

# Cosine distance (1 - cosine similarity). Results above this are dropped.
MAX_DISTANCE = float(os.getenv("RAG_MAX_DISTANCE", "0.8"))

# --- uploads ---------------------------------------------------------------
MAX_FILE_SIZE = 50 * 1024 * 1024
MAX_FILES_PER_REQUEST = 10
ALLOWED_EXTENSIONS = frozenset({".pdf", ".txt", ".docx", ".eml"})

# --- link following --------------------------------------------------------
MAX_FOLLOW_LINKS = int(os.getenv("RAG_MAX_FOLLOW_LINKS", "20"))
FOLLOW_CONCURRENCY = int(os.getenv("RAG_FOLLOW_CONCURRENCY", "3"))
FOLLOW_DEADLINE_S = float(os.getenv("RAG_FOLLOW_DEADLINE_S", "120"))
FOLLOW_DELAY_S = float(os.getenv("RAG_FOLLOW_DELAY_S", "0.2"))

# --- outbound safety -------------------------------------------------------
# Optional allowlist of domains (or subdomains) the scraper may reach. Empty
# means "no allowlist": the private-range check still applies, but any public
# host is reachable. Strongly recommended when the service is exposed.
ALLOWED_CRAWL_DOMAINS = _csv_set(os.getenv("RAG_ALLOWED_CRAWL_DOMAINS", ""))

# How many redirect hops _validate_crawl_url will follow while re-validating.
MAX_REDIRECTS = int(os.getenv("RAG_MAX_REDIRECTS", "5"))

PROBE_TTL_S = 300
