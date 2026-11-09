"""Scraper factory: Crawl4AI (self-hosted) with Firecrawl (cloud) fallback.

A third engine, `direct`, lives in `rag/direct.py` and is reachable only by
asking for it by name. It is deliberately not in `_AUTO_CHAIN`: adding it there
would change which engine every existing caller lands on, which is a behaviour
change disguised as a feature. `tests/test_direct.py` pins the chain.

Key behavioural change from the original single-file version: **redirects are
not followed here**. The SSRF guard in `security.validate_crawl_url()` walks
the redirect chain and hands over an already-validated final URL. If this layer
followed redirects independently, that validation would be decorative.

For Crawl4AI this is a request to *its* `/crawl` endpoint, which follows
redirects server-side where we cannot intercept them. That is why the crawler
container's egress must also be restricted at the network level — noted in the
README and enforced by the allowlist when `RAG_ALLOWED_CRAWL_DOMAINS` is set.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from urllib.parse import urlparse

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from . import config

logger = logging.getLogger("rag.scrapers")


def safe_url(url: str) -> str:
    """URL without query string / userinfo, for logs and error messages."""
    p = urlparse(url)
    return f"{p.scheme}://{p.hostname or ''}{p.path}"


def hrefs(items) -> list[str]:
    """Extract href strings from a list of strings / dicts."""
    out: list[str] = []
    if not isinstance(items, list):
        return out
    for item in items:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict) and item.get("href"):
            out.append(str(item["href"]))
    return [x for x in out if x]


def is_transient(exc: BaseException) -> bool:
    """True for network-level errors worth retrying."""
    transient: tuple[type[BaseException], ...] = (
        httpx.TimeoutException,
        httpx.ConnectError,
        TimeoutError,
        ConnectionError,
    )
    try:
        import requests  # firecrawl SDK is requests-based

        transient += (requests.exceptions.Timeout, requests.exceptions.ConnectionError)
    except ImportError:
        pass
    return isinstance(exc, transient)


class ScraperError(Exception):
    """Raised when a scraping engine fails.

    `public_detail` is the part of the reason that is safe and useful to hand
    back to a caller over HTTP. The distinction matters: `cause` may carry
    upstream text (a Firecrawl error body, a crawler's own message) that we do
    not want to reflect to a third party, while a gate measurement or an HTTP
    status is exactly what the caller needs in order to act.

    It exists because the extraction gate in `rag.direct` raises with measured
    numbers on purpose -- "extracted only 172 chars from 577 bytes" tells a
    caller to retry with a browser engine, where "scraping failed" tells it
    nothing. Discarding that at the HTTP boundary defeated the gate's whole
    reason for raising.
    """

    def __init__(
        self,
        url: str,
        engine: str,
        cause: str | Exception,
        retryable: bool = False,
        public_detail: str | None = None,
    ):
        self.url = url
        self.engine = engine
        self.cause = cause
        self.retryable = retryable
        # Default to the cause when it is a string the engine author wrote.
        # Exception instances are never reflected: their text comes from
        # somewhere we do not control.
        if public_detail is None and isinstance(cause, str):
            public_detail = cause
        self.public_detail = public_detail
        super().__init__(f"[{engine}] {safe_url(url)}: {cause}")


@dataclass
class ScrapeResult:
    markdown: str
    links: list[str] = field(default_factory=list)
    source_url: str = ""
    engine: str = ""


def _log_retry(rs) -> None:
    url = rs.args[1] if len(rs.args) > 1 else "?"
    logger.warning(
        "Retry %d for %s after: %s",
        rs.attempt_number,
        safe_url(url) if url != "?" else url,
        rs.outcome.exception() if rs.outcome else "?",
    )


def _scraper_retry():
    return retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception(lambda e: isinstance(e, ScraperError) and e.retryable),
        before_sleep=_log_retry,
        reraise=True,
    )


class ScraperFactory:
    """Unified async scraper: Crawl4AI first, Firecrawl as fallback."""

    _AUTO_CHAIN: list[str] = ["crawl4ai", "firecrawl"]

    def __init__(self, *, allow_redirects: bool = False) -> None:
        self._fc = None
        if config.FIRECRAWL_API_KEY:
            from firecrawl import V1FirecrawlApp

            self._fc = V1FirecrawlApp(api_key=config.FIRECRAWL_API_KEY)
        self._client = httpx.AsyncClient(timeout=60.0, follow_redirects=allow_redirects)
        self._probe_results: dict[str, bool] = {}
        self._probe_time: float = 0.0
        logger.info(
            "ScraperFactory initialised (firecrawl=%s, crawl4ai=%s, redirects=%s)",
            "yes" if self._fc else "no",
            "yes" if config.CRAWL_URL else "no",
            allow_redirects,
        )

    async def __aenter__(self) -> ScraperFactory:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self._client.aclose()
        logger.info("ScraperFactory closed.")

    # -- headers -----------------------------------------------------------
    @staticmethod
    def _crawl4ai_headers() -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if config.CRAWL_TOKEN:
            headers["Authorization"] = f"Bearer {config.CRAWL_TOKEN}"
        return headers

    # -- health probes -----------------------------------------------------
    async def probe_engines(self) -> dict[str, bool]:
        results = {"crawl4ai": False, "firecrawl": False}

        if config.CRAWL_URL:
            try:
                base = config.CRAWL_URL.rsplit("/", 1)[0]
                resp = await self._client.get(
                    f"{base}/health",
                    headers=self._crawl4ai_headers(),
                    timeout=10.0,
                )
                results["crawl4ai"] = resp.status_code == 200
            except Exception as exc:
                logger.warning("Crawl4AI probe failed: %s", exc)
        else:
            logger.warning("CRAWL_URL not configured; Crawl4AI unavailable.")

        if self._fc is not None:
            try:
                resp = await asyncio.to_thread(self._fc.get_credit_usage)
                results["firecrawl"] = resp is not None
            except Exception as exc:
                logger.warning("Firecrawl probe failed: %s", exc)
        else:
            logger.warning("FIRECRAWL_API_KEY not configured; Firecrawl unavailable.")

        self._probe_results = results
        self._probe_time = time.monotonic()
        logger.info(
            "Engine probe: crawl4ai=%s, firecrawl=%s",
            "OK" if results["crawl4ai"] else "UNAVAILABLE",
            "OK" if results["firecrawl"] else "UNAVAILABLE",
        )
        return results

    async def get_probe_results(self, max_age: float | None = None) -> dict[str, bool]:
        """Public accessor; re-probes if the cached result is stale.

        `max_age` is resolved at call time, not bound as a default argument, so
        monkeypatching `config.PROBE_TTL_S` in a test actually takes effect.
        """
        if max_age is None:
            max_age = config.PROBE_TTL_S
        if time.monotonic() - self._probe_time > max_age:
            await self.probe_engines()
        return dict(self._probe_results)

    def cached_probe_results(self) -> dict[str, bool]:
        """Last known probe result. Never performs network I/O."""
        return dict(self._probe_results)

    # -- Firecrawl ---------------------------------------------------------
    @_scraper_retry()
    async def _fetch_firecrawl(self, url: str) -> ScrapeResult:
        if self._fc is None:
            raise ScraperError(url, "firecrawl", "FIRECRAWL_API_KEY not configured")

        logger.info("Fetching %s via Firecrawl", safe_url(url))
        try:
            result = await asyncio.to_thread(
                self._fc.scrape_url,
                url,
                formats=["markdown", "links"],
                timeout=30_000,
            )
        except Exception as exc:
            raise ScraperError(url, "firecrawl", exc, retryable=is_transient(exc)) from exc

        if not result:
            raise ScraperError(url, "firecrawl", "empty response")

        if hasattr(result, "success") and not result.success:
            err = getattr(result, "error", "unknown")
            raise ScraperError(url, "firecrawl", f"scrape failed: {err}")

        data = (
            result.get("data", result)
            if isinstance(result, dict)
            else getattr(result, "data", result)
        )

        if isinstance(data, dict):
            md = data.get("markdown")
            raw_links = data.get("links", [])
        else:
            md = getattr(data, "markdown", None)
            raw_links = getattr(data, "links", [])

        if isinstance(md, list):
            md = "\n\n".join(md)
        if not md:
            raise ScraperError(url, "firecrawl", "no markdown in response")

        links = hrefs(raw_links)
        logger.info("Firecrawl: got %d links from %s", len(links), safe_url(url))
        return ScrapeResult(
            markdown=md[: config.MAX_SCRAPE_CHARS],
            links=links,
            source_url=url,
            engine="firecrawl",
        )

    # -- Crawl4AI ----------------------------------------------------------
    @_scraper_retry()
    async def _fetch_crawl4ai(self, url: str) -> ScrapeResult:
        if not config.CRAWL_URL:
            raise ScraperError(url, "crawl4ai", "CRAWL_URL not configured")

        logger.info("Fetching %s via Crawl4AI", safe_url(url))
        payload: dict = {"urls": [url]}

        try:
            response = await self._client.post(
                config.CRAWL_URL, json=payload, headers=self._crawl4ai_headers()
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            raise ScraperError(
                url, "crawl4ai", f"http {code}", retryable=(code == 429 or code >= 500)
            ) from exc
        except httpx.TimeoutException as exc:
            raise ScraperError(url, "crawl4ai", f"timeout: {exc}", retryable=True) from exc
        except httpx.ConnectError as exc:
            raise ScraperError(url, "crawl4ai", f"connect error: {exc}", retryable=True) from exc
        except httpx.HTTPError as exc:
            raise ScraperError(url, "crawl4ai", f"http error: {exc}") from exc
        except ValueError as exc:
            raise ScraperError(url, "crawl4ai", f"invalid JSON: {exc}") from exc

        return self._extract_markdown_from_crawl4ai(data, url)

    @staticmethod
    def _extract_markdown_from_crawl4ai(crawl_data: dict, url: str) -> ScrapeResult:
        results = crawl_data.get("results")
        if not isinstance(results, list) or not results:
            raise ScraperError(url, "crawl4ai", "crawler returned no results")

        first = results[0]
        if not isinstance(first, dict):
            raise ScraperError(url, "crawl4ai", "invalid result structure")

        if first.get("success") is False:
            raise ScraperError(url, "crawl4ai", f"crawler error: {first.get('error', 'unknown')}")

        md = first.get("markdown", "")
        if isinstance(md, dict):
            md = md.get("raw_markdown") or md.get("markdown") or md.get("fit_markdown") or ""
        if isinstance(md, list):
            md = "\n\n".join(md)
        if not isinstance(md, str):
            raise ScraperError(url, "crawl4ai", f"unexpected markdown type: {type(md).__name__}")

        md = md.strip()
        if not md:
            raise ScraperError(url, "crawl4ai", "no markdown content returned")

        raw_links = first.get("links", [])
        links: list[str] = []
        if isinstance(raw_links, list):
            links = hrefs(raw_links)
        elif isinstance(raw_links, dict):
            for key in ("internal", "external"):
                sub = raw_links.get(key, [])
                if isinstance(sub, str):
                    links.append(sub)
                else:
                    links.extend(hrefs(sub))
            links = [x for x in links if x]

        if not links:
            logger.debug(
                "Crawl4AI returned no parseable links for %s (type=%s)",
                safe_url(url),
                type(raw_links).__name__,
            )

        logger.info("Crawl4AI: got %d links from %s", len(links), safe_url(url))
        return ScrapeResult(
            markdown=md[: config.MAX_SCRAPE_CHARS],
            links=links,
            source_url=url,
            engine="crawl4ai",
        )

    # -- Public API with fallback ------------------------------------------
    async def get_content(self, url: str, engine: str = "auto") -> ScrapeResult:
        engine = engine.lower()

        if engine == "firecrawl":
            return await self._fetch_firecrawl(url)
        if engine == "crawl4ai":
            return await self._fetch_crawl4ai(url)
        if engine == "direct":
            # Deferred import: rag.direct imports ScraperError, ScrapeResult,
            # hrefs and safe_url from this module, so a module-level import
            # here would be circular. The in-function form costs nothing after
            # the first call and keeps the dependency one-way on paper.
            from .direct import fetch_direct

            return await fetch_direct(self._client, url)
        if engine != "auto":
            raise ValueError(
                f"Unsupported engine: {engine!r}. Use 'crawl4ai', 'firecrawl', 'direct', or 'auto'."
            )

        last_error: Exception | None = None
        for eng in self._AUTO_CHAIN:
            if eng == "firecrawl" and self._fc is None:
                continue
            if eng == "crawl4ai" and not config.CRAWL_URL:
                continue
            try:
                logger.info("[auto] Trying %s for %s", eng, safe_url(url))
                if eng == "crawl4ai":
                    return await self._fetch_crawl4ai(url)
                return await self._fetch_firecrawl(url)
            except ScraperError as exc:
                last_error = exc
                logger.warning("[auto] %s failed (%s); trying next engine.", eng, exc)

        raise ScraperError(
            url, "auto", f"All engines failed. Last error: {last_error}"
        ) from last_error
