"""Entrypoint shim.

The application lives in the `rag` package. This module exists so the existing
container command (`uvicorn main:app`) keeps working unchanged, and so anyone
familiar with the old layout is not surprised by a missing `main.py`.

Module map:

    rag/config.py     every tunable, read once from the environment
    rag/security.py   API-key auth + redirect-aware SSRF validation
    rag/scrapers.py   Crawl4AI / Firecrawl, redirects deliberately disabled
    rag/store.py      LanceDB: embed, atomic ingest, retrieval queries
    rag/documents.py  file loaders and chunking
    rag/api.py        HTTP routes
    rag/app.py        FastAPI factory and lifespan
    rag/state.py      breaks the api <-> app import cycle
"""

from __future__ import annotations

from rag.app import create_app

app = create_app()

__all__ = ["app"]
