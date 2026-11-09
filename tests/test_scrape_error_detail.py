"""Tests for the scrape-failure detail in the 502 body.

The extraction gate in `rag.direct` raises with measured numbers on purpose:
"extracted only 172 chars from 577 bytes" tells a caller to retry with a browser
engine, while "scraping failed" tells it nothing it can act on. `rag/api.py`
logged that reason and replaced it with a flat string, so the gate's whole
justification was defeated at the HTTP boundary.

These call the **real route** (`ingest_url`) with only the scraper stubbed.
An earlier version of this file re-implemented the except-branch inside the
test, which would have passed happily while the route stayed broken -- the
mapping is the thing under test, so the test has to run it.
"""

from __future__ import annotations

import asyncio
import types

import pytest
from fastapi import HTTPException

from rag import api, scrapers, security, state


class _FailingFactory:
    """Stands in for ScraperFactory; raises whatever it is given."""

    def __init__(self, error: Exception):
        self.error = error
        self.calls: list[str] = []

    async def get_content(self, url: str, engine: str = "auto"):
        self.calls.append(url)
        raise self.error


def _ingest(factory_owner: _FailingFactory, **body_kwargs) -> HTTPException:
    """Drive `ingest_url` and return the HTTPException it raised.

    Only the SSRF probe and the app state are substituted. The route body --
    engine validation, the fetch call, the except-branch and the shape of the
    detail -- all run for real.
    """
    app = types.SimpleNamespace(state=types.SimpleNamespace(scraper=factory_owner))

    async def passthrough(url: str, **kwargs):
        return security.ValidatedUrl(requested=url, final=url)

    original_validate = api.security.validate_crawl_url
    original_state = state.state.app
    api.security.validate_crawl_url = passthrough
    state.state.bind(app)
    try:
        body = api.IngestUrlRequest(**body_kwargs)
        with pytest.raises(HTTPException) as exc:
            asyncio.run(api.ingest_url(body))
        return exc.value
    finally:
        api.security.validate_crawl_url = original_validate
        # Restore through `bind`, not by assigning to `state.state.app`.
        # `bind` writes the *class* attribute; assigning to the instance would
        # create a shadowing instance attribute that survives the restore and
        # makes every later test read `None`.
        state.state.bind(original_state)


# ---------------------------------------------------------------------------
# public_detail on the exception
# ---------------------------------------------------------------------------


def test_a_string_cause_becomes_the_public_detail():
    """Engine-authored strings are the whole point: they are written to be read."""
    exc = scrapers.ScraperError("https://e/x", "direct", "extracted only 172 chars from 577 bytes")
    assert exc.public_detail == "extracted only 172 chars from 577 bytes"


def test_an_exception_cause_is_not_reflected():
    """Upstream text must not reach a third-party caller.

    A Firecrawl error body or a crawler's own message is written by someone
    else, for an audience that is not our caller. Returning `str(cause)` would
    turn this endpoint into a reflector.
    """
    exc = scrapers.ScraperError("https://e/x", "firecrawl", RuntimeError("upstream says SECRET"))
    assert exc.public_detail is None
    assert "SECRET" not in str(exc.public_detail or "")


def test_public_detail_can_be_set_explicitly_for_an_exception_cause():
    """An engine may classify an exception and supply its own safe wording."""
    exc = scrapers.ScraperError(
        "https://e/x", "direct", TimeoutError("boom"), public_detail="timed out"
    )
    assert exc.public_detail == "timed out"


# ---------------------------------------------------------------------------
# the 502 body, through the real route
# ---------------------------------------------------------------------------


def test_gate_measurements_reach_the_body():
    """The regression. Before this, the reason was logged and discarded."""
    exc = _ingest(
        _FailingFactory(
            scrapers.ScraperError(
                "https://example.com/",
                "direct",
                "extracted only 172 chars from 577 bytes; the page looks client-rendered",
            )
        ),
        url="https://example.com/",
        engine="direct",
    )

    assert exc.status_code == 502
    assert "172 chars" in exc.detail["reason"]
    # The generic message must still be there for callers that read it.
    assert exc.detail["detail"] == "Scraping failed for the requested URL."
    assert exc.detail["reason"] != exc.detail["detail"]


def test_engine_is_reported_so_the_caller_can_fall_back():
    """Knowing it was `direct` is what makes "retry with crawl4ai" actionable."""
    exc = _ingest(
        _FailingFactory(
            scrapers.ScraperError(
                "https://example.com/f.pdf",
                "direct",
                "unsupported content type 'application/pdf'",
            )
        ),
        url="https://example.com/f.pdf",
        engine="direct",
    )
    assert exc.detail["engine"] == "direct"
    assert "application/pdf" in exc.detail["reason"]


def test_retryable_is_carried_through():
    """A 429 must not be presented as a permanent failure."""
    error = scrapers.ScraperError("https://e/x", "crawl4ai", "http 429", retryable=True)
    exc = _ingest(_FailingFactory(error), url="https://e/x", engine="crawl4ai")
    assert exc.detail["retryable"] is True


def test_reason_is_null_not_missing_when_withheld():
    """A withheld reason must be an explicit null, not a dropped key.

    The tool distinguishes "the server explained itself but had nothing safe to
    say" from "this server predates the field"; a missing key makes those the
    same thing, and the tool would then have to guess.
    """
    exc = _ingest(
        _FailingFactory(
            scrapers.ScraperError("https://e/x", "firecrawl", RuntimeError("internal detail"))
        ),
        url="https://e/x",
        engine="firecrawl",
    )
    assert "reason" in exc.detail
    assert exc.detail["reason"] is None


def test_the_scraper_is_actually_invoked():
    """Guards against the test passing on a route that never reached the fetch."""
    factory = _FailingFactory(scrapers.ScraperError("https://e/x", "direct", "boom"))
    _ingest(factory, url="https://e/x", engine="direct")
    assert factory.calls == ["https://e/x"]


# ---------------------------------------------------------------------------
# ordering and wiring
# ---------------------------------------------------------------------------


def test_engine_validation_happens_before_any_scrape():
    """An unknown engine is a 400, not a 502 — and costs no fetch.

    If validation moved after the fetch, a typo would trigger a network call
    and report itself as a scraping failure.
    """
    factory = _FailingFactory(scrapers.ScraperError("https://e/x", "auto", "unused"))
    exc = _ingest(factory, url="https://e/x", engine="nonsense")

    assert exc.status_code == 400
    assert factory.calls == []


def test_api_module_maps_the_scraper_error_it_imports():
    """Cheap guard against the mapping silently targeting the wrong class."""
    assert api.scrapers.ScraperError is scrapers.ScraperError
