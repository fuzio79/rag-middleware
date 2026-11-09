"""Direct HTTP fetcher: validate and fetch in one pass.

Why this is not in ``ScraperFactory._AUTO_CHAIN``
------------------------------------------------
``auto`` is ``crawl4ai -> firecrawl``. Inserting a direct fetcher anywhere in
that chain would change which engine a given URL lands on -- and therefore what
gets stored and what gets billed -- for every existing caller. It is an explicit
opt-in (``engine="direct"``) until there is evidence it belongs in the chain.
``tests/test_direct.py`` pins the chain so the scoping cannot drift silently.

Why this lives in its own module
--------------------------------
The property worth having here is that **the code that validates a URL is the
code that fetches it**. ``security.validate_crawl_url()`` is honestly described
as a pre-filter, because Crawl4AI follows redirects and runs JavaScript where we
cannot observe it. A fetch we perform ourselves can re-validate each hop
immediately before requesting it, in the same loop, so nothing is fetched that
was not checked and nothing is checked twice. Keeping that loop in one small
module makes the property auditable.

Why the gate exists
-------------------
A plain fetch of a client-rendered page returns an empty mount point plus nav
and footer. Reporting that as ``ingested, 3 chunks`` is the same class of defect
as a confident ``unchanged`` over an empty table: silent, wrong, and successful
looking. The gate therefore **raises** with the measurements attached, so the
caller can fall back to a browser engine deliberately instead of indexing
boilerplate.

What this engine deliberately does not do
-----------------------------------------
No JavaScript execution, no cookie handling, no content filtering comparable to
Crawl4AI's ``fit_markdown``. A page that needs those is a Crawl4AI page; this
engine's job is to be exact and free on the pages that do not.
"""

from __future__ import annotations

import logging
import re
import time
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx
from fastapi import HTTPException

from . import config, security
from .scrapers import ScraperError, ScrapeResult, _scraper_retry, hrefs, safe_url

logger = logging.getLogger("rag.direct")

# SKIP: never visible text. `noscript` is skipped because its contents are an
# error message about the JavaScript that did not run, not page content.
_SKIP_TAGS = frozenset({"script", "style", "template", "svg", "noscript"})

# BLOCK: emit a paragraph break. Without these the extractor produces one long
# line and the character splitter has no structure to work with, which shows up
# as worse retrieval rather than as a visible bug.
_BLOCK_TAGS = frozenset(
    {
        "address",
        "article",
        "aside",
        "blockquote",
        "br",
        "dd",
        "div",
        "dl",
        "dt",
        "figcaption",
        "figure",
        "footer",
        "form",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "tbody",
        "td",
        "tfoot",
        "th",
        "thead",
        "tr",
        "ul",
    }
)

_PASSTHROUGH_TYPES = frozenset({"text/markdown", "text/x-markdown", "text/plain"})
_HTML_TYPES = frozenset({"text/html", "application/xhtml+xml"})

_ROBOTS_MAX_BYTES = 500_000
_robots_cache: dict[str, tuple[float, RobotFileParser | None]] = {}


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------


class _Extractor(HTMLParser):
    """Flatten HTML to paragraph-separated text, recording link text.

    ``link_text`` is measured rather than estimated: a page whose visible text
    is mostly anchor text is a navigation page, and indexing it produces chunks
    that match every query a little and none of them well.
    """

    def __init__(self) -> None:
        # convert_charrefs defaults to True, so handle_data already receives
        # decoded text. Do not unescape again -- that was a real defect in the
        # .eml loader, where a literal `&amp;lt;` became `<`.
        super().__init__()
        self.parts: list[str] = []
        self.links: list[str] = []
        self._skip = 0
        self._link = 0
        self.link_text = 0

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self._skip += 1
            return
        if tag == "a":
            self._link += 1
            href = dict(attrs).get("href")
            if href:
                self.links.append(str(href))
        if tag in _BLOCK_TAGS:
            self.parts.append("\n\n")

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS:
            if self._skip:
                self._skip -= 1
            return
        if tag == "a" and self._link:
            self._link -= 1
        if tag in _BLOCK_TAGS:
            self.parts.append("\n\n")

    def handle_startendtag(self, tag, attrs):
        # `<br/>` and `<hr/>` never get an end tag; without this the paragraph
        # break is skipped for self-closing block elements.
        if tag in _BLOCK_TAGS:
            self.parts.append("\n\n")

    def handle_data(self, data: str):
        if self._skip:
            return
        text = data.strip()
        if not text:
            return
        self.parts.append(text + " ")
        if self._link:
            self.link_text += len(" ".join(text.split()))


def _normalise(raw: str) -> str:
    """Collapse whitespace but keep paragraph boundaries."""
    paragraphs: list[str] = []
    current: list[str] = []
    for line in raw.split("\n"):
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            current.append(line)
        elif current:
            paragraphs.append(" ".join(current))
            current = []
    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(paragraphs)


def _media_type(content_type_header: str) -> str:
    return content_type_header.split(";")[0].strip().lower()


def _charset(content_type_header: str) -> str | None:
    for param in content_type_header.split(";")[1:]:
        name, _, value = param.partition("=")
        if name.strip().lower() == "charset":
            value = value.strip().strip('"').strip("'")
            if value:
                return value
    return None


# ---------------------------------------------------------------------------
# the quality gate
# ---------------------------------------------------------------------------


def _gate(url: str, text: str, raw_bytes: int, link_text: int) -> None:
    """Refuse to report success over a page that has no content in it.

    Every message carries the measurements, because "empty content" is not
    actionable and is exactly how this class of bug survives: the caller cannot
    tell a genuinely empty page from a JavaScript shell from a nav-only index.
    """
    text_len = len(text)
    if text_len < config.DIRECT_MIN_TEXT_CHARS:
        raise ScraperError(
            url,
            "direct",
            (
                f"extracted only {text_len} chars from {raw_bytes} bytes; the page "
                "looks client-rendered or empty (this engine does not run JavaScript)"
            ),
        )

    ratio = text_len / max(raw_bytes, 1)
    if ratio < config.DIRECT_MIN_TEXT_RATIO:
        raise ScraperError(
            url,
            "direct",
            (
                f"extracted {text_len} chars from {raw_bytes} bytes "
                f"({ratio:.1%} text); the page looks client-rendered or is mostly markup"
            ),
        )

    density = link_text / max(text_len, 1)
    if density > config.DIRECT_MAX_LINK_DENSITY:
        raise ScraperError(
            url,
            "direct",
            (
                f"{density:.0%} of the extracted text is link text "
                f"({link_text}/{text_len} chars); this looks like a navigation or "
                "index page rather than an article"
            ),
        )


# ---------------------------------------------------------------------------
# robots.txt
# ---------------------------------------------------------------------------


def _request_headers() -> dict[str, str]:
    return {
        "User-Agent": config.DIRECT_USER_AGENT,
        "Accept": "text/html,text/plain;q=0.9,*/*;q=0.5",
    }


async def _robots_for(client: httpx.AsyncClient, url: str) -> RobotFileParser | None:
    """Parsed robots.txt for this origin, or None when there is nothing to honour.

    Cached per origin because link following can touch dozens of pages on one
    site and re-fetching robots.txt for each would be both slow and rude to the
    server whose rules we are trying to respect.
    """
    parsed = urlparse(url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    now = time.monotonic()
    cached = _robots_cache.get(origin)
    if cached is not None and now - cached[0] < config.ROBOTS_CACHE_TTL_S:
        return cached[1]

    parser: RobotFileParser | None = None
    robots_url = f"{origin}/robots.txt"
    try:
        # robots.txt is fetched from the same origin, so it goes through the
        # same guard. A host we would refuse to crawl is a host we do not ask
        # for permission either.
        await security._validate_single(robots_url)
        async with client.stream("GET", robots_url, headers=_request_headers()) as resp:
            if resp.status_code < 400:
                body = await _read_capped(resp, _ROBOTS_MAX_BYTES)
                parser = RobotFileParser()
                parser.parse(body.decode("utf-8", errors="replace").splitlines())
            elif resp.status_code >= 500:
                # RFC 9309 says an unavailable robots.txt should be treated as
                # a full disallow. Refusing every ingest because an origin
                # returned one 503 is worse than the risk it guards against, so
                # this allows and says so loudly. Stated, not assumed away.
                logger.warning(
                    "robots.txt for %s answered %d; allowing the fetch",
                    origin,
                    resp.status_code,
                )
            # 4xx: no robots.txt means no restrictions (RFC 9309 2.3.1.3).
    except (httpx.HTTPError, HTTPException, OSError, ValueError) as exc:
        logger.warning("could not read robots.txt for %s (%s); allowing", origin, exc)

    _robots_cache[origin] = (now, parser)
    return parser


async def _robots_allows(client: httpx.AsyncClient, url: str) -> bool:
    parser = await _robots_for(client, url)
    if parser is None:
        return True
    return parser.can_fetch(config.DIRECT_USER_AGENT, url)


# ---------------------------------------------------------------------------
# body reading
# ---------------------------------------------------------------------------


async def _read_capped(resp: httpx.Response, limit: int) -> bytes:
    """Read at most ``limit`` bytes, stopping rather than truncating afterwards.

    ``MAX_SCRAPE_CHARS`` bounds the stored text; this bounds the download. They
    are different quantities -- a page can be tens of megabytes of markup that
    truncates to a few hundred kilobytes of text -- and reading the whole body
    before truncating would make the cap decorative.
    """
    blocks: list[bytes] = []
    total = 0
    async for block in resp.aiter_bytes():
        if not block:
            continue
        remaining = limit - total
        if remaining <= 0:
            break
        blocks.append(block[:remaining])
        total += min(len(block), remaining)
    return b"".join(blocks)


# ---------------------------------------------------------------------------
# result construction
# ---------------------------------------------------------------------------


def _build_result(final_url: str, content_type_header: str, body: bytes) -> ScrapeResult:
    media = _media_type(content_type_header)
    charset = _charset(content_type_header)

    if not media:
        # Some servers omit Content-Type entirely. Sniff the unambiguous case
        # instead of refusing a page that is plainly HTML; anything else keeps
        # the named refusal below.
        head = body[:2048].lower()
        media = "text/html" if b"<html" in head or b"<!doctype html" in head else ""

    if media in _HTML_TYPES:
        raw = body.decode(charset or "utf-8", errors="replace")
        extractor = _Extractor()
        extractor.feed(raw)
        text = _normalise("".join(extractor.parts))
        _gate(final_url, text, len(body), extractor.link_text)
        return ScrapeResult(
            markdown=text[: config.MAX_SCRAPE_CHARS],
            links=hrefs(extractor.links),
            source_url=final_url,
            engine="direct",
        )

    if media in _PASSTHROUGH_TYPES:
        text = body.decode(charset or "utf-8", errors="replace").strip()
        if not text:
            raise ScraperError(final_url, "direct", f"empty body ({len(body)} bytes)")
        return ScrapeResult(
            markdown=text[: config.MAX_SCRAPE_CHARS],
            links=[],
            source_url=final_url,
            engine="direct",
        )

    raise ScraperError(
        final_url,
        "direct",
        (
            f"unsupported content type {media or 'unknown'!r}; "
            "the direct engine handles text/html, text/markdown and text/plain"
        ),
    )


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------


@_scraper_retry()
async def fetch_direct(client: httpx.AsyncClient, url: str) -> ScrapeResult:
    """Validate and fetch in one pass.

    ``client`` must be the factory's shared client, built with
    ``follow_redirects=False``. This loop is the only thing that follows a hop,
    which is what lets each one be validated before it is requested: validate,
    then fetch, then read ``Location`` and validate the next hop.

    The alternative -- call ``validate_crawl_url()`` and then fetch separately --
    issues a second request to a response that may differ, and would make the
    direct engine a validate-then-refetch rather than the one property that
    justifies it.
    """
    current = url

    for _ in range(config.MAX_REDIRECTS):
        await security._validate_single(current)

        if config.RESPECT_ROBOTS and not await _robots_allows(client, current):
            raise ScraperError(current, "direct", "disallowed by robots.txt (RAG_RESPECT_ROBOTS)")

        try:
            async with client.stream("GET", current, headers=_request_headers()) as resp:
                status = resp.status_code

                if status in security.REDIRECT_CODES:
                    location = resp.headers.get("location")
                    if not location:
                        raise ScraperError(
                            current, "direct", f"http {status} without a Location header"
                        )
                    current = urljoin(current, location)
                    continue

                if status >= 400:
                    raise ScraperError(
                        current,
                        "direct",
                        f"http {status}",
                        retryable=(status == 429 or status >= 500),
                    )

                body = await _read_capped(resp, config.DIRECT_MAX_DOWNLOAD_BYTES)
                result = _build_result(current, resp.headers.get("content-type", ""), body)
                logger.info(
                    "direct: %d chars, %d links from %s",
                    len(result.markdown),
                    len(result.links),
                    safe_url(current),
                )
                return result
        except httpx.TimeoutException as exc:
            raise ScraperError(current, "direct", f"timeout: {exc}", retryable=True) from exc
        except httpx.ConnectError as exc:
            raise ScraperError(current, "direct", f"connect error: {exc}", retryable=True) from exc
        except httpx.HTTPError as exc:
            raise ScraperError(current, "direct", f"http error: {exc}") from exc

    raise ScraperError(url, "direct", "too many redirects while fetching")
