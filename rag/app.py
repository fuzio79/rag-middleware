"""Application factory and lifespan wiring."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from starlette.responses import JSONResponse

from . import api, config, scrapers, security, state, store

logger = logging.getLogger("rag.app")


class DocsGate:
    """Require the API key for `/docs`, `/redoc` and `/openapi.json`.

    FastAPI mounts these three routes itself, outside any `APIRouter`, so the
    router-level `Depends(require_api_key)` never reaches them -- which made the
    README's "every route except /health requires a key" false. The schema is
    not a secret, but it enumerates the ingest surface to anything that can
    reach the port.

    Implemented as ASGI middleware rather than `@app.middleware("http")` so the
    check runs before routing, and so it cannot be sidestepped by a path that
    differs only in a trailing slash.

    Exceptions are rendered here rather than allowed to propagate: this sits
    *outside* Starlette's ExceptionMiddleware, so a raised HTTPException would
    reach the server as a 500 instead of the intended 401/503.
    """

    _DOC_PATHS = frozenset({"/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"})

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] == "http":
            path = scope.get("path", "").rstrip("/") or "/"
            if path in self._DOC_PATHS:
                key = None
                headers = scope.get("headers") or []
                for name, value in headers:
                    if name.lower() == b"x-api-key":
                        key = value.decode("latin-1")
                        break
                try:
                    await security.require_api_key(key)
                except HTTPException as exc:
                    response = JSONResponse(
                        {"detail": exc.detail},
                        status_code=exc.status_code,
                        headers=exc.headers,
                    )
                    await response(scope, receive, send)
                    return
        await self.app(scope, receive, send)


@asynccontextmanager
async def lifespan(app: FastAPI):
    store.init_embeddings()
    store.init_table()

    factory = scrapers.ScraperFactory()
    async with factory:
        await factory.probe_engines()
        app.state.scraper = factory

        if config.REQUIRE_AUTH and not config.API_KEY:
            logger.warning(
                "RAG_REQUIRE_AUTH is on but RAG_API_KEY is empty. Every data "
                "endpoint will answer 503 until a key is set. Set RAG_API_KEY, "
                "or set RAG_REQUIRE_AUTH=false to run unauthenticated on a "
                "trusted network."
            )
        elif not config.REQUIRE_AUTH:
            logger.warning(
                "Authentication is DISABLED (RAG_REQUIRE_AUTH=false). Every "
                "endpoint is open. Only acceptable on a trusted network."
            )

        if config.PUBLIC_DOCS:
            logger.warning(
                "RAG_PUBLIC_DOCS=true: /docs, /redoc and /openapi.json are "
                "served without a key, advertising the ingest surface."
            )

        yield

    store.reset_embeddings()
    store.reset_table()


def create_app() -> FastAPI:
    app = FastAPI(title="RAG Knowledge Base Tool", lifespan=lifespan)
    app.include_router(api.router)
    app.include_router(api.open_router)
    if not config.PUBLIC_DOCS:
        # Added before state.bind so the gate is in place for every request the
        # app ever serves, including the first ones during startup.
        app.add_middleware(DocsGate)
    state.state.bind(app)
    return app
