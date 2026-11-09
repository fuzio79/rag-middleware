# rag-middleware

A small, self-hosted RAG (Retrieval-Augmented Generation) service for
**Open WebUI**. It scrapes web pages and indexes documents into a local
**LanceDB** vector store, then serves retrieval queries as Open WebUI tool
calls.

Everything runs on your own hardware. Embeddings are computed on CPU with
`sentence-transformers`, so no GPU is required.

## What it does

- **Scrape** a URL to clean markdown, via [Crawl4AI](https://github.com/unclecode/crawl4ai)
  (self-hosted) with [Firecrawl](https://firecrawl.dev) as an optional fallback,
  or over plain HTTP with `engine="direct"` (no browser, no JavaScript).
- **Upload** `.pdf`, `.txt`, `.docx` and `.eml` files.
- **Chunk and embed** them on CPU, storing vectors in LanceDB.
- **Retrieve** relevant passages for a query, wrapped in an untrusted-data
  envelope so the model does not treat page content as instructions.
- **Read a page without storing it** — fetch, chunk, embed in memory and return
  only the passages that match a question, so a single fact from a long document
  costs a few paragraphs instead of the whole page.
- **Delete** everything stored for one source, so an ingest can be undone.
- **Expose** it as Open WebUI tool calls: `rag_search`, `ingest_url`,
  `fetch_url` and `delete_source`.

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

Read one page without keeping it:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/rag/fetch \
  -H "X-API-Key: $RAG_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://www.rfc-editor.org/rfc/rfc9309","query":"what does robots.txt allow"}'
```

That returns `"stored": false` and a `context` holding the few passages that
match the query. It is the right call when the page answers one question and
should not join the knowledge base. Measured on a 24 000-character RFC: 3
passages, 2 500 characters returned, ~480 ms.

Remove what an ingest added:

```bash
# Rehearse first: reports how many chunks would go, changes nothing.
curl -X POST http://127.0.0.1:8000/api/v1/rag/delete \
  -H "X-API-Key: $RAG_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"source":"https://en.wikipedia.org/wiki/Vector_database","dry_run":true}'

# Then for real. `source` accepts the raw URL, a `url:`-prefixed key, or a filename.
```

Deleting claims every alias a source may have been stored under, so a row
written by an older build is removed too. It is irrecoverable and removes one
named source — there is no delete-by-query.

## Endpoints

| Method | Path | Auth | Purpose |
|---|---|---|---|
| `POST` | `/api/v1/rag/search` | `X-API-Key` | Retrieve passages for a query |
| `POST` | `/api/v1/rag/ingest` | `X-API-Key` | Scrape a URL and index it |
| `POST` | `/api/v1/rag/fetch` | `X-API-Key` | Read a URL and answer a query from it, **writing nothing** |
| `POST` | `/api/v1/rag/delete` | `X-API-Key` | Remove every stored chunk for one source |
| `POST` | `/api/v1/knowledge/upload_files/` | `X-API-Key` | Upload and index files |
| `GET` | `/health` | none | Liveness plus last known engine reachability |
| `GET` | `/docs`, `/redoc`, `/openapi.json` | `X-API-Key` | Schema and interactive docs |

FastAPI mounts the docs routes itself, outside the application routers, so they
do not inherit the key dependency; they are gated explicitly instead. Set
`RAG_PUBLIC_DOCS=true` to serve them openly, which is only appropriate for
local development — the schema advertises the ingest surface to anything that
can reach the port.

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
| `RAG_PUBLIC_DOCS` | `false` | Serve `/docs`, `/redoc` and `/openapi.json` without a key |
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
| `RAG_MAX_FOLLOW_LINKS` | `20` | Cap on followed links per ingest |
| `RAG_FOLLOW_CONCURRENCY` | `3` | Parallel scrapes when following links |
| `RAG_FOLLOW_DELAY_S` | `0.2` | Delay between follow-up scrapes |
| `RAG_FOLLOW_DEADLINE_S` | `120` | Time budget for the follow phase |
| `RAG_ALLOWED_CRAWL_DOMAINS` | *(empty)* | Optional narrowing allowlist. Empty means public-only: any non-local host is crawlable |
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
- **The SSRF guard is a pre-filter, not a boundary.** `validate_crawl_url()`
  resolves and range-checks the submitted URL *and* every redirect hop it
  observes, probing with GET (the method the crawler uses) and failing closed
  when a hop cannot be checked. It also refuses loopback, link-local, private
  and reserved ranges. What it **cannot** do: a hostile server can answer the
  probe one way and the crawler another (JavaScript redirects, `meta refresh`),
  and DNS can change between the check and the crawler's own lookup. Do not
  rely on it as the only control.
- **The default policy is public-only.** With `RAG_ALLOWED_CRAWL_DOMAINS`
  empty, every globally routable host may be crawled and loopback, private,
  link-local and reserved ranges are refused. That is "anything that is not
  local", and it is the intended configuration for a LAN deployment.
  `RAG_ALLOWED_CRAWL_DOMAINS` only *narrows* that set further; an allowlist is
  not what provides the private-range refusal.
- **The control that actually binds the crawler is egress policy.** Crawl4AI is
  a browser running in its own container; it follows redirects and loads
  subresources where this service cannot see them. Restricting where that
  container can reach at the network level is the mitigation that does not
  depend on this code seeing the request.
- **Indexed content is untrusted.** Retrieval wraps passages in a per-response
  random delimiter so indexed text cannot close the envelope and impersonate
  the caller. Prompt injection through indexed pages remains possible
  regardless of framing — only index sources you trust.
- **Uploads are capped**, at 50 MB per file and 10 files per request, with an
  extension allowlist (`.pdf`, `.txt`, `.docx`, `.eml`).
- **Run a single worker.** Write serialisation is a per-process lock, so
  `uvicorn --workers N` reintroduces duplicate rows and loads the embedding
  model N times. `scripts/preflight.sh` checks this and the other
  deployment-level items.

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
| `rag/security.py` | API-key auth and redirect-aware SSRF pre-filter |
| `rag/scrapers.py` | Crawl4AI / Firecrawl, redirects deliberately disabled |
| `rag/store.py` | LanceDB: embed, serialised ingest, retrieval |
| `rag/documents.py` | file loaders and chunking |
| `rag/api.py` | HTTP routes |
| `rag/app.py` | FastAPI factory, lifespan, docs gate |
| `rag/state.py` | breaks the api ↔ app import cycle |

Run the tests and the linter:

```bash
python -m pytest -q
ruff check .
ruff format --check .
mypy
```

Use `python -m pytest`, not the bare `pytest` script: only the `-m` form puts
the working directory on `sys.path`, which the `from rag import ...` imports in
the tests depend on. The GitHub workflow runs it this way for the same reason.

Helper scripts:

| Script | Purpose |
|---|---|
| `scripts/build.sh` | Build the image |
| `scripts/rebuild.sh` | Rebuild the image (never restarts containers) |
| `scripts/test-scrap.sh` | Smoke-test health, ingest and search |
| `scripts/upload.sh` | Bulk-upload a directory of documents |
| `scripts/clean.sh` | Drop the vector table (requires `--yes-i-know`) |
| `scripts/preflight.sh` | Report deployment-level risks (worker count, crawler egress, allowlist, auth, docs gate, table health) |

The smoke-test and upload scripts read `RAG_HOST` and `RAG_API_KEY` from the
environment.

`requirements.txt` is fully pinned and the base image is digest-pinned in the
`Dockerfile`, so a rebuild reproduces the same dependency set.

## Publishing to the public repository

Development happens on a private Gitea instance. The public repo is a
**curated snapshot**, pushed by a `workflow_dispatch` workflow. Publishing is a
decision, never a side effect of a merge.

### What is published

Exactly what `git archive` emits, with `.gitattributes export-ignore` deciding
the exclusions. Currently 36 files.

| Excluded | Why |
|---|---|
| `plan.md`, `*.internal.md` | internal topology, hostnames, security decisions |
| `.gitea/` | Gitea CI; useless off that instance |
| `.gitignore`, `.dockerignore` | instance-specific |

`.github/` is deliberately **kept** — the public repo runs its own Actions
(`ci.yml` for lint/type/test, `release.yml` to publish a prebuilt image to
GHCR on a `v*.*.*` tag).

### Why the snapshot has no history

`git archive` into a fresh `git init` produces an **orphan commit**. Gitea
history is never pushed, because history cannot be un-published. That also
means a secret committed and later removed is still gone from the public repo.

### The workflow refuses to publish when

- `plan.md` or `.gitea/` reached the snapshot (export-ignore regression)
- `.github/workflows/ci.yml` is missing (public repo would have no CI)
- **gitleaks** finds anything in the snapshot — scanned with `--redact` so a
  finding is not echoed into the log
- the push is not confirmed by typing `PUBLISH`; anything else is a dry run

After pushing it re-reads the remote and expects `404` for `plan.md`.

#### These gates were verified by making them fail, not by reading them

A gate nobody has seen reject anything is a comment. Each was exercised against
a real archive of `main`:

| Gate | Test | Result |
|---|---|---|
| `plan.md` absent | planted a `plan.md` in the snapshot | refuses to publish |
| `.gitea/` absent | (same mechanism, `export-ignore`) | absent in the archive |
| `.github/workflows/ci.yml` present | present in the archive | passes |
| gitleaks | planted a randomly-generated AWS key pair | `exit 1`, "leaks found" |

That gitleaks test is worth describing, because the first attempt **passed when
it should have failed**: the keys were AWS's own documented examples
(`AKIAIOSFODNN7EXAMPLE`), which gitleaks allowlists deliberately. A random key
of the same shape was caught immediately. If a future scan comes back clean on
something that looks like a credential, check whether it is a known-example
value before believing the result.

Also note what gitleaks does **not** gate: `plan.md` is not secret-shaped, so it
would pass a secret scan. Its protection is `export-ignore` alone, which is why
the explicit `plan.md` assertion exists alongside the secret scan and why both
must keep running.

### The credential: `OCG_GITHUB_PAT`

A fine-grained GitHub PAT with `contents:write` on the public repo, held as a
Gitea organisation secret. The name is prefixed `OCG_` because Gitea rejects
secret names beginning with `GITHUB` or `GITEA`.

This token is the only genuinely sensitive thing in the publish path. What it
can and cannot reach:

**Not at risk of publication**

- It lives in the Gitea secret store, not in any file. `.env` is gitignored and
  export-ignored, and the workflow reads the value via
  `${{ secrets.OCG_GITHUB_PAT }}`.
- It is passed to `git remote add` as a URL credential inside an ephemeral job
  container that is removed when the job ends.
- gitleaks runs over the snapshot *before* the push and fails closed.

**Residual risks, in order of likelihood**

1. **The remote URL is written into `.git/config`** inside the job container.
   That container is disposable and never published, so this only matters if
   someone execs into a live job and reads it.
2. **A `set -x` or a failing command could echo the URL.** The workflow does
   not enable shell tracing, and `set -eu` without `-x` means commands are not
   printed. If `set -x` is ever added to the Publish step, the token appears in
   the log.
3. **The Gitea Actions log itself.** Step output is stored on the Gitea
   instance. A leaked value there is visible to anyone who can read that
   repository's Actions logs.
4. **Gitea substitutes an empty string for an unreadable secret.** A missing
   PAT therefore looks like an empty variable, which is why the Preflight step
   checks for it explicitly rather than letting the push fail confusingly.

**What the token does not do:** it authenticates the *push*. It grants nothing
to the published content, and it cannot be recovered from the snapshot because
it is never written to a file in it.

**Blast radius if it did leak.** A fine-grained token with `contents:write` on
one repository allows rewriting that repository's history. It does **not**
grant access to other repositories, packages, or account settings — provided it
really is fine-grained and scoped to this single repository, which is what the
workflow's comment assumes. A classic PAT with `repo` scope would be a much
larger problem: that scope reads and writes every repository the account can
reach.

**If you suspect exposure:** revoke the PAT in GitHub settings first, then
rotate the Gitea organisation secret. The workflow picks up the new value on its
next run; no code change is needed.

### Running it

Actions → `public-sync` → Run workflow:

1. `confirm: dry-run`, `create_private_repo: true` — clones, applies
   export-ignore, runs both structural gates and gitleaks, prints the file
   list, pushes nothing.
2. Review the file list and the gitleaks result.
3. Re-run with `confirm: PUBLISH`.

Repository creation is **private** first. Flipping it public is a separate,
manual act, and is the point at which `release.yml` starts publishing images to
a world-readable registry on every version tag.

## License

MIT. See [LICENSE](LICENSE).
