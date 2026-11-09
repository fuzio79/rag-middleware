"""Configuration, read once from the environment at import time.

Every tunable lives here so nothing is hidden inside an endpoint module. Tests
import this and monkeypatch individual attributes rather than re-reading env.

Test authors can monkeypatch any attribute here and have it take effect.
Functions that used to bind these values as default arguments
(`retrieve_context`, `get_probe_results`) now take `None` and resolve it
internally, so the monkeypatch reaches them.

One exception remains, by design: pydantic `Field(default=config.X)` in the
request models is bound at import. Those are the HTTP contract and are
exercised through the client anyway, so patching `config` will not change a
request model's default. Pass the field explicitly in such a test.
"""

from __future__ import annotations

import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag")


def _csv_set(value: str) -> frozenset[str]:
    return frozenset(x.strip().lower() for x in value.split(",") if x.strip())


def _flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes", "on"}


# --- storage ---------------------------------------------------------------
DB_PATH = os.getenv("RAG_DB_PATH", "local_knowledge_base")
EMBEDDING_MODEL_NAME = os.getenv("RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# --- auth ------------------------------------------------------------------
# Shared API key. When unset the service still starts (so a misconfigured
# deployment is debuggable) but every protected endpoint refuses and a loud
# warning is logged at startup. Set RAG_REQUIRE_AUTH=false to opt out
# explicitly, which is only appropriate on a trusted network.
API_KEY = os.getenv("RAG_API_KEY", "")
REQUIRE_AUTH = _flag("RAG_REQUIRE_AUTH", "true")

# FastAPI serves /docs, /redoc and /openapi.json outside the routers, so they do
# not inherit the API-key dependency. They are gated deliberately (see
# app.DocsGate) because the schema advertises the ingest surface to anything
# that can reach the port. Opt out only for local development.
PUBLIC_DOCS = _flag("RAG_PUBLIC_DOCS", "false")

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
# Default policy is public-only: with this empty, any globally routable host is
# reachable and loopback/private/link-local/reserved ranges are refused by
# validate_crawl_url. Setting it NARROWS that set; it is not what provides the
# private-range refusal. The control that does not depend on this code seeing
# the request is egress policy on the crawl4ai container.
ALLOWED_CRAWL_DOMAINS = _csv_set(os.getenv("RAG_ALLOWED_CRAWL_DOMAINS", ""))

# How many redirect hops _validate_crawl_url will follow while re-validating.
MAX_REDIRECTS = int(os.getenv("RAG_MAX_REDIRECTS", "5"))

PROBE_TTL_S = 300

# --- direct fetcher --------------------------------------------------------
# `engine="direct"` fetches with httpx instead of handing the URL to a browser.
# Not part of ScraperFactory._AUTO_CHAIN: inserting it there would change which
# engine a URL lands on, and therefore what gets stored and billed, for every
# existing caller. Explicit opt-in until there is evidence it belongs.

# Honour robots.txt. On by default; the flag exists for operator-controlled
# mirrors and for a site whose robots.txt you own, not as a convenience.
RESPECT_ROBOTS = _flag("RAG_RESPECT_ROBOTS", "true")
# robots.txt is cached per origin. Link following can touch dozens of pages on
# one site, and re-fetching the file for each would be rude to the server whose
# rules we are trying to respect.
ROBOTS_CACHE_TTL_S = float(os.getenv("RAG_ROBOTS_CACHE_TTL_S", "3600"))

# Identifies this service to the sites it fetches. A crawler that lies about
# what it is has no standing to complain about being blocked.
DIRECT_USER_AGENT = os.getenv(
    "RAG_DIRECT_USER_AGENT", "rag-middleware/1.0 (self-hosted RAG; +robots.txt honoured)"
)

# Bounds the DOWNLOAD, which is a different quantity from MAX_SCRAPE_CHARS
# (which bounds the text stored). A page can be 20 MB of markup that yields a
# few hundred KB of text, and reading it all before truncating would make the
# text cap decorative.
DIRECT_MAX_DOWNLOAD_BYTES = int(os.getenv("RAG_DIRECT_MAX_DOWNLOAD_BYTES", "5000000"))

# --- extraction quality gate ----------------------------------------------
# A plain fetch of a client-rendered page yields an empty mount point plus nav
# and footer. Indexing that and reporting success is the same class of defect as
# a confident `unchanged` over an empty table, so the gate RAISES instead.
#
# These three defaults are PROPOSALS pending calibration against real pages
# (see issue #28). They are the mechanism, not a measured contract: raise
# DIRECT_MIN_TEXT_CHARS and the gate rejects short legitimate pages; lower it
# and a JavaScript shell slips through.
DIRECT_MIN_TEXT_CHARS = int(os.getenv("RAG_DIRECT_MIN_TEXT_CHARS", "500"))
# Extracted text as a fraction of the raw body. Catches a big document whose
# visible text is negligible relative to its markup.
DIRECT_MIN_TEXT_RATIO = float(os.getenv("RAG_DIRECT_MIN_TEXT_RATIO", "0.01"))
# Fraction of the extracted text that is anchor text. Catches navigation and
# index pages, which produce chunks that match every query weakly.
DIRECT_MAX_LINK_DENSITY = float(os.getenv("RAG_DIRECT_MAX_LINK_DENSITY", "0.6"))
