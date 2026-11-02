"""Application factory and lifespan wiring."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from . import api, config, scrapers, state, store

logger = logging.getLogger("rag.app")


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

        yield

    store.reset_embeddings()
    store.reset_table()


def create_app() -> FastAPI:
    app = FastAPI(title="RAG Knowledge Base Tool", lifespan=lifespan)
    app.include_router(api.router)
    app.include_router(api.open_router)
    state.state.bind(app)
    return app
