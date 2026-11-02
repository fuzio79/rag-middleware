# rag-middleware

A small, self-hosted RAG (Retrieval-Augmented Generation) service for
**Open WebUI**. It scrapes web pages and indexes documents into a local
**LanceDB** vector store, then serves retrieval queries as Open WebUI tool
calls.

Everything runs on your own hardware. Embeddings are computed on CPU with
`sentence-transformers`, so no GPU is required.

## What it does

- **Scrape** a URL to clean markdown, via [Crawl4AI](https://github.com/unclecode/crawl4ai)
  (self-hosted) with [Firecrawl](https://firecrawl.dev) as an optional fallback.
- **Upload** `.pdf`, `.txt`, `.docx` and `.eml` files.
- **Chunk and embed** them on CPU, storing vectors in LanceDB.
- **Retrieve** relevant passages for a query, wrapped in an untrusted-data
  envelope so the model does not treat page content as instructions.
- **Expose** it all as Open WebUI tool calls: `rag_search` and `ingest_url`.

Optionally follow same-site links from a page to ingest a whole site section.

## Architecture

```
Open WebUI ──> rag-middleware (FastAPI) ──> LanceDB (vector store)
                     │
                     ├──> Crawl4AI   (primary scraper, self-hosted)
                     └──> Firecrawl  (optional fallback, cloud)
```

| Piece | Choice |
|---|---|
| Embedding model | `all-MiniLM-L6-v2` (CPU, 384-dim) |
| Vector store | LanceDB, embedded (file-based) |
| Chunking | `RecursiveCharacterTextSplitter`, 800 chars / 100 overlap |
| Similarity | cosine distance, results above `RAG_MAX_DISTANCE` are dropped |

Chunks are capped at 800 characters because `all-MiniLM-L6-v2` truncates at
256 tokens (~1000 characters); longer chunks would silently lose their tail.

## Quick start

```bash
git clone https://github.com/fuzio79/rag-middleware.git
cd rag-middleware
cp .env.example .env
```

Generate an API key and put it in `.env`:

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

Then start the service:

```bash
docker compose up -d
```

Check it is alive. `/health` is deliberately unauthenticated so container
healthchecks work:

```bash
curl -s http://127.0.0.1:8000/health
# {"status":"online","engines":{"crawl4ai":true,"firecrawl":false}}
```

Every *data* endpoint requires the key, sent as `X-API-Key`:

```bash
export RAG_API_KEY='...your key...'
```

Index something:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/rag/ingest \
  -H "X-API-Key: $RAG_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://en.wikipedia.org/wiki/Vector_database"}'
```

Query it:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/rag/search \
  -H "X-API-Key: $RAG_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"query":"what is a vector database"}'
```

## Endpoints

| Method | Path | Auth | Purpose |
|---|---|---|---|
| `POST` | `/api/v1/rag/search` | `X-API-Key` | Retrieve passages for a query |
| `POST` | `/api/v1/rag/ingest` | `X-API-Key` | Scrape a URL and index it |
| `POST` | `/api/v1/knowledge/upload_files/` | `X-API-Key` | Upload and index files |
| `GET` | `/health` | none | Liveness plus last known engine reachability |
| `GET` | `/openapi.json` | none | OpenAPI schema |

Interactive docs at `/docs`.

## Authentication

Authentication is **on by default** and **fail-closed**:

| Server state | Result |
|---|---|
| `RAG_API_KEY` set | data endpoints require a matching `X-API-Key`, else `401` |
| `RAG_API_KEY` unset, `RAG_REQUIRE_AUTH=true` | data endpoints return `503` and refuse to serve |
| `RAG_REQUIRE_AUTH=false` | data endpoints are open — trusted networks only |

`503` rather than "serve openly" is intentional: a missing secret is a
misconfiguration, and failing closed is the safer default. `/health` never
requires a key, and never makes an outbound call (so it cannot incur Firecrawl
charges).

## Configuration

All configuration is via environment variables. See `.env.example` for a
ready-to-copy template.

| Variable | Default | Purpose |
|---|---|---|
| `RAG_API_KEY` | *(empty)* | Shared API key; required unless auth is disabled |
| `RAG_REQUIRE_AUTH` | `true` | Set `false` to serve without auth (trusted networks only) |
| `RAG_DB_PATH` | `local_knowledge_base` | LanceDB directory |
| `CRAWL_URL` | `http://crawl4ai:11235/crawl` | Crawl4AI endpoint |
| `CRAWL4AI_API_TOKEN` | *(empty)* | Bearer token for Crawl4AI |
| `FIRECRAWL_API_KEY` | *(empty)* | Optional Firecrawl fallback key |
| `RAG_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model |
| `RAG_CHUNK_SIZE` | `800` | Characters per chunk |
| `RAG_CHUNK_OVERLAP` | `100` | Chunk overlap |
| `RAG_MAX_DISTANCE` | `0.8` | Cosine-distance cutoff for relevance |
| `RAG_MAX_SCRAPE_CHARS` | `500000` | Truncation limit per scraped page |
| `RAG_MAX_CHUNKS_PER_REQUEST` | `2000` | Hard ceiling on chunks embedded per request |
| `RAG_MAX_FOLLOW_LINKS` | `30` | Cap on followed links per ingest |
| `RAG_FOLLOW_CONCURRENCY` | `3` | Parallel scrapes when following links |
| `RAG_FOLLOW_DELAY_S` | `0.2` | Delay between follow-up scrapes |
| `RAG_FOLLOW_DEADLINE_S` | `120` | Time budget for the follow phase |
| `RAG_ALLOWED_CRAWL_DOMAINS` | *(empty)* | Comma-separated egress allowlist |
| `RAG_MAX_REDIRECTS` | `5` | Redirect hops re-validated during ingest |

`RAG_MAX_CHUNKS_PER_REQUEST` spans the main page **and** every followed link.
Without it, `max_follow` times an unbounded page is unbounded CPU work from a
single request.

### About the two scrapers

Crawl4AI is tried first. If it fails, `engine="auto"` **silently** falls back
to Firecrawl, so a broken Crawl4AI looks like a working system. To check each
engine independently, inside the container:

```bash
docker compose exec rag-service python3 -c "
import asyncio
from rag.scrapers import ScraperFactory
async def go():
    async with ScraperFactory() as f:
        print(await f.probe_engines())
asyncio.run(go())"
```

To prove a credential rather than just liveness, scrape with each engine
explicitly:

```bash
docker compose exec rag-service python3 -c "
import asyncio
from rag.scrapers import ScraperFactory
async def go():
    async with ScraperFactory() as f:
        for e in ('crawl4ai','firecrawl'):
            try:
                r = await f.get_content('https://example.com', engine=e)
                print(e, r.engine, len(r.markdown or ''))
            except Exception as exc:
                print(e, 'FAILED', exc)
asyncio.run(go())"
```

## Open WebUI integration

The tool plugin is maintained in a separate repository, not here. That is the
copy Open WebUI loads; there is deliberately no duplicate in this repo.

1. Admin Panel → Workspace → Tools → `+` → paste the tool file → Save.
2. Set the `rag_server_host` valve to your service URL (e.g.
   `http://localhost:8000`) and the `api_key` valve to your `RAG_API_KEY`.
3. Enable the tool on a model that supports function calling.

The tool exposes `rag_search(query)` and `ingest_url(url, engine)`.

## Security notes

Read this before exposing the service beyond localhost.

- **Authentication is a single shared key.** Every caller uses the same
  `X-API-Key`. There are no per-user credentials and no rate limiting; put a
  reverse proxy in front of it if you need either. An open
  `/api/v1/rag/ingest` lets anyone make your host fetch arbitrary URLs and
  spend CPU embedding them.
- **SSRF guard covers redirects.** `validate_crawl_url()` resolves and
  range-checks the submitted URL *and* every redirect hop, up to
  `RAG_MAX_REDIRECTS`; scrapers are constructed with redirects disabled so
  that validated address is the one actually used. Private, loopback,
  link-local and reserved ranges are rejected. DNS rebinding is blunted but
  not eliminated — resolve-then-connect is still two steps.
- **Crawl4AI still follows redirects server-side.** This service validates
  before handing a URL over, but the crawler container itself is a separate
  process. Set `RAG_ALLOWED_CRAWL_DOMAINS` and restrict the crawler's egress
  at the network level.
- **The allowlist is off by default.** Empty `RAG_ALLOWED_CRAWL_DOMAINS` means
  any public host can be crawled. Set it for any non-trivial deployment.
- **Indexed content is untrusted.** Retrieved passages are wrapped in an
  explicit untrusted-data envelope, but prompt injection through indexed pages
  remains possible. Only index sources you trust.
- **Uploads are capped**, at 50 MB per file and 10 files per request, with an
  extension allowlist (`.pdf`, `.txt`, `.docx`, `.eml`).

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn main:app --reload
```

`main.py` is a thin entrypoint shim; the application lives in the `rag/`
package:

| Module | Responsibility |
|---|---|
| `rag/config.py` | every tunable, read once from the environment |
| `rag/security.py` | API-key auth and redirect-aware SSRF validation |
| `rag/scrapers.py` | Crawl4AI / Firecrawl, redirects deliberately disabled |
| `rag/store.py` | LanceDB: embed, atomic ingest, retrieval |
| `rag/documents.py` | file loaders and chunking |
| `rag/api.py` | HTTP routes |
| `rag/app.py` | FastAPI factory and lifespan |
| `rag/state.py` | breaks the api ↔ app import cycle |

Run the tests and the linter:

```bash
pytest -q
ruff check .
ruff format --check .
```

Helper scripts:

| Script | Purpose |
|---|---|
| `scripts/build.sh` | Build the image |
| `scripts/rebuild.sh` | Rebuild the image (never restarts containers) |
| `scripts/test-scrap.sh` | Smoke-test health, ingest and search |
| `scripts/upload.sh` | Bulk-upload a directory of documents |
| `scripts/clean.sh` | Drop the vector table (requires `--yes-i-know`) |

The smoke-test and upload scripts read `RAG_HOST` and `RAG_API_KEY` from the
environment.

`requirements.txt` is fully pinned and the base image is digest-pinned in the
`Dockerfile`, so a rebuild reproduces the same dependency set.

## License

MIT. See [LICENSE](LICENSE).
