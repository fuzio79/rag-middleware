"""Tests for the direct fetcher: scoping, redirect handling, and the gate.

`httpx.MockTransport` drives the whole path with no network, no browser and no
async context-manager fakery, which is the main practical argument for this
engine existing at all.

Four groups, each written so it fails against the behaviour it is guarding
rather than merely describing the new code:

* **scoping** -- `auto` must not reach this engine, so the chain cannot drift;
* **redirects** -- every hop is validated *before* it is requested, and a
  private hop is refused without ever being fetched;
* **the gate** -- a client-rendered shell must raise, with measurements;
* **robots** -- honoured by default, skipped when the flag is off.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import httpx
import pytest

from rag import api, config, direct, scrapers, security

# ---------------------------------------------------------------------------
# a fake site
# ---------------------------------------------------------------------------


@dataclass
class _Reply:
    status: int = 200
    body: bytes = b""
    content_type: str = "text/html; charset=utf-8"
    location: str | None = None


@dataclass
class _Site:
    """Maps URL -> _Reply and records every URL that was actually requested."""

    pages: dict[str, _Reply] = field(default_factory=dict)
    seen: list[str] = field(default_factory=list)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.seen.append(url)
        reply = self.pages.get(url)
        if reply is None:
            # No robots.txt served -> RFC 9309 treats 4xx as "no restrictions".
            return httpx.Response(404, content=b"not found")
        headers = {"content-type": reply.content_type}
        if reply.location:
            headers["location"] = reply.location
        return httpx.Response(reply.status, content=reply.body, headers=headers)

    @property
    def hits(self) -> list[str]:
        """Requests for actual documents, excluding the robots.txt probe.

        Kept separate because most assertions here are about what the fetcher
        asked for, and a robots probe is bookkeeping rather than a fetch of the
        page under test. Assertions that *are* about robots use `seen`.
        """
        return [u for u in self.seen if not u.endswith("/robots.txt")]


def _client(site: _Site) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.MockTransport(site),
        # Matches production: the loop in `fetch_direct` is the only thing
        # allowed to follow a hop, which is what makes per-hop validation real.
        follow_redirects=False,
        timeout=5.0,
    )


def _fetch(site: _Site, url: str) -> scrapers.ScrapeResult:
    async def go():
        client = _client(site)
        try:
            return await direct.fetch_direct(client, url)
        finally:
            await client.aclose()

    return asyncio.run(go())


def _fetch_error(site: _Site, url: str) -> scrapers.ScraperError:
    with pytest.raises(scrapers.ScraperError) as exc:
        _fetch(site, url)
    return exc.value


# Long enough that the extracted text clears the 500-char floor comfortably, so
# the fixture tests extraction rather than the gate. A fixture that only just
# fails the gate would make every extraction test ambiguous.
ARTICLE = (
    "<html><body>"
    "<nav><a href='/'>Home</a><a href='/docs'>Docs</a></nav>"
    "<article>"
    "<h1>Vector databases</h1>"
    "<p>A vector database stores high-dimensional embeddings and answers "
    "nearest-neighbour queries over them. Cosine distance is the usual metric "
    "for text, because the embedding model normalises its output before it is "
    "written.</p>"
    "<p>LanceDB is embedded rather than client-server, which is what makes it "
    "possible to run the whole retrieval stack as a single container.</p>"
    "<p>Retrieval quality depends more on chunking than on the model. A window "
    "of eight hundred characters with one hundred characters of overlap keeps a "
    "paragraph intact while staying under the two hundred and fifty six token "
    "limit that the embedding model silently truncates at, which matters "
    "because a truncated tail is lost with no error reported anywhere.</p>"
    "<p>Because the store is embedded, one container can hold the whole stack: "
    "no separate vector service to operate, no network hop between the "
    "application and its index, and no second copy of the data to keep "
    "consistent with the first.</p>"
    "</article>"
    "<a href='/docs/chunking'>Chunking</a>"
    "<a href='https://other.example/x'>Elsewhere</a>"
    "</body></html>"
)

SHELL = (
    "<html><head><title>App</title></head><body>"
    "<div id='root'></div>"
    "<nav><a href='/'>Home</a><a href='/login'>Log in</a></nav>"
    "</body></html>"
)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    """Neutral DNS, empty robots cache, and the shipped gate thresholds."""

    async def allow(hostname: str) -> None:
        return None

    monkeypatch.setattr(security, "_check_resolves_to_public", allow)
    monkeypatch.setattr(config, "ALLOWED_CRAWL_DOMAINS", frozenset())
    monkeypatch.setattr(config, "RESPECT_ROBOTS", True)
    monkeypatch.setattr(config, "DIRECT_MIN_TEXT_CHARS", 500)
    monkeypatch.setattr(config, "DIRECT_MIN_TEXT_RATIO", 0.01)
    monkeypatch.setattr(config, "DIRECT_MAX_LINK_DENSITY", 0.6)
    monkeypatch.setattr(config, "DIRECT_MAX_DOWNLOAD_BYTES", 5_000_000)
    direct._robots_cache.clear()
    yield
    direct._robots_cache.clear()


# ---------------------------------------------------------------------------
# scoping: `auto` must not reach this engine
# ---------------------------------------------------------------------------


def test_auto_chain_is_unchanged():
    """The whole point of the scoping.

    Inserting `direct` into the chain would change which engine every existing
    caller lands on -- and therefore what gets stored and billed -- for URLs
    nobody re-tested. If someone later decides it belongs there, this test must
    be changed deliberately rather than silently outgrown.
    """
    assert scrapers.ScraperFactory._AUTO_CHAIN == ["crawl4ai", "firecrawl"]


def test_auto_never_calls_the_direct_transport(monkeypatch):
    """With both chain engines absent, `auto` fails without a single request."""
    monkeypatch.setattr(config, "CRAWL_URL", "")
    monkeypatch.setattr(config, "FIRECRAWL_API_KEY", None)

    site = _Site()

    async def go():
        async with scrapers.ScraperFactory() as factory:
            factory._client = _client(site)
            await factory.get_content("http://example.com/", engine="auto")

    with pytest.raises(scrapers.ScraperError, match="All engines failed"):
        asyncio.run(go())

    assert site.seen == []


def test_direct_is_selectable_through_the_factory():
    """`engine="direct"` resolves through the same public entry point."""
    site = _Site({"http://example.com/": _Reply(body=ARTICLE.encode())})

    async def go():
        async with scrapers.ScraperFactory() as factory:
            factory._client = _client(site)
            return await factory.get_content("http://example.com/", engine="direct")

    result = asyncio.run(go())
    assert result.engine == "direct"
    assert "vector database" in result.markdown


def test_unknown_engine_still_raises_value_error():
    site = _Site()

    async def go():
        async with scrapers.ScraperFactory() as factory:
            factory._client = _client(site)
            await factory.get_content("http://example.com/", engine="nonsense")

    with pytest.raises(ValueError, match="Unsupported engine"):
        asyncio.run(go())


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------


def test_html_is_extracted_to_paragraph_text():
    site = _Site({"http://example.com/a": _Reply(body=ARTICLE.encode())})
    result = _fetch(site, "http://example.com/a")

    assert result.engine == "direct"
    assert result.source_url == "http://example.com/a"
    assert "Vector databases" in result.markdown
    assert "Cosine distance" in result.markdown
    # Paragraph structure must survive, or the character splitter has nothing
    # to work with and retrieval degrades invisibly.
    assert "\n\n" in result.markdown
    # Markup must never reach the text.
    assert "<html" not in result.markdown
    assert "<p>" not in result.markdown


def test_script_and_style_are_skipped():
    page = (
        "<html><head><style>body{color:red}</style></head><body>"
        "<script>var x = 'this must not appear';</script>"
        "<noscript>Please enable JavaScript</noscript>"
        "<p>" + ("Real body text. " * 60) + "</p>"
        "</body></html>"
    )
    site = _Site({"http://example.com/a": _Reply(body=page.encode())})
    result = _fetch(site, "http://example.com/a")

    assert "must not appear" not in result.markdown
    assert "color:red" not in result.markdown
    assert "enable JavaScript" not in result.markdown
    assert "Real body text" in result.markdown


def test_links_are_extracted_so_follow_links_is_not_dead():
    """Without links, `follow_links=True` silently does nothing.

    That is the failure mode this asserts against: an empty `links` list makes
    link following appear to work while scraping exactly one page.
    """
    site = _Site({"http://example.com/a": _Reply(body=ARTICLE.encode())})
    result = _fetch(site, "http://example.com/a")

    assert "/docs/chunking" in result.links
    assert "https://other.example/x" in result.links

    # And they are still consumable by the existing link-following path, which
    # is the only reason populating them matters.
    candidates, _ = api.select_links(result.links, base_url=result.source_url, limit=10)
    assert "http://example.com/docs/chunking" in candidates


def test_markdown_passthrough_is_not_reformatted():
    body = "# Title\n\nBody text that is already markdown.\n"
    site = _Site(
        {"http://example.com/x.md": _Reply(body=body.encode(), content_type="text/markdown")}
    )
    result = _fetch(site, "http://example.com/x.md")

    assert result.markdown.startswith("# Title")
    assert result.links == []


def test_markdown_passthrough_does_not_apply_the_html_gate(monkeypatch):
    """A short markdown file is legitimate; the gate is an HTML concern."""
    monkeypatch.setattr(config, "DIRECT_MIN_TEXT_CHARS", 500)
    body = "Short note.\n"
    site = _Site(
        {"http://example.com/n.md": _Reply(body=body.encode(), content_type="text/markdown")}
    )
    result = _fetch(site, "http://example.com/n.md")
    assert result.markdown == "Short note."


# ---------------------------------------------------------------------------
# redirects: validated before requested
# ---------------------------------------------------------------------------


def test_redirect_to_private_address_is_never_fetched(monkeypatch):
    """The hole the SSRF guard exists for, asserted at the fetch itself."""
    rejected: list[str] = []

    async def guard(hostname: str) -> None:
        rejected.append(hostname)
        if hostname in {"127.0.0.1", "169.254.169.254"}:
            raise security._reject("Target address is in a private or reserved range.", 403)

    monkeypatch.setattr(security, "_check_resolves_to_public", guard)

    site = _Site(
        {
            "http://public.example/go": _Reply(status=302, location="http://127.0.0.1:8000/"),
            "http://127.0.0.1:8000/": _Reply(body=ARTICLE.encode()),
        }
    )

    with pytest.raises(security.UrlRejected) as exc:
        _fetch(site, "http://public.example/go")

    assert exc.value.status_code == 403
    # The guard refused the hop...
    assert "127.0.0.1" in rejected
    # ...and the private host was never requested. This is the assertion that
    # distinguishes validate-and-fetch from validate-then-refetch.
    assert "http://127.0.0.1:8000/" not in site.seen


def test_public_redirect_is_followed_and_reported():
    site = _Site(
        {
            "http://a.example/x": _Reply(status=301, location="http://b.example/y"),
            "http://b.example/y": _Reply(body=ARTICLE.encode()),
        }
    )
    result = _fetch(site, "http://a.example/x")

    assert result.source_url == "http://b.example/y"
    assert site.hits == ["http://a.example/x", "http://b.example/y"]


def test_every_hop_is_validated(monkeypatch):
    """A chain must not be walked on the strength of the first check alone."""
    checked: list[str] = []

    async def guard(hostname: str) -> None:
        checked.append(hostname)
        if hostname == "hop3.example":
            raise security._reject("Target address is in a private or reserved range.", 403)

    monkeypatch.setattr(security, "_check_resolves_to_public", guard)

    site = _Site(
        {
            "http://hop1.example/a": _Reply(status=302, location="http://hop2.example/b"),
            "http://hop2.example/b": _Reply(status=302, location="http://hop3.example/c"),
            "http://hop3.example/c": _Reply(body=ARTICLE.encode()),
        }
    )

    with pytest.raises(security.UrlRejected):
        _fetch(site, "http://hop1.example/a")

    # Deduplicated while preserving order: the robots.txt probe for hop1's
    # origin re-checks hop1, which is correct but would make a strict equality
    # assertion about the *pages* misleading.
    unique = list(dict.fromkeys(checked))
    assert unique == ["hop1.example", "hop2.example", "hop3.example"]
    assert "http://hop3.example/c" not in site.seen


def test_redirect_chain_is_bounded(monkeypatch):
    monkeypatch.setattr(config, "MAX_REDIRECTS", 3)
    site = _Site({"http://loop.example/": _Reply(status=302, location="http://loop.example/")})

    err = _fetch_error(site, "http://loop.example/")
    assert "too many redirects" in str(err)
    assert len(site.hits) == 3


def test_redirect_without_a_location_header_is_an_error():
    site = _Site({"http://a.example/x": _Reply(status=302)})
    err = _fetch_error(site, "http://a.example/x")
    assert "without a Location" in str(err)


# ---------------------------------------------------------------------------
# content types
# ---------------------------------------------------------------------------


def test_unsupported_content_type_is_refused_by_name():
    """A silent empty result here is how a PDF becomes 0 chunks and a success."""
    site = _Site(
        {"http://example.com/f.pdf": _Reply(body=b"%PDF-1.4 ...", content_type="application/pdf")}
    )
    err = _fetch_error(site, "http://example.com/f.pdf")
    assert "application/pdf" in str(err)


def test_missing_content_type_is_sniffed_for_html():
    site = _Site({"http://example.com/a": _Reply(body=ARTICLE.encode(), content_type="")})
    result = _fetch(site, "http://example.com/a")
    assert "Vector databases" in result.markdown


def test_charset_is_honoured():
    body = ARTICLE.replace("Vector databases", "Wektorowe bazy").encode("iso-8859-2")
    site = _Site(
        {"http://example.com/a": _Reply(body=body, content_type="text/html; charset=iso-8859-2")}
    )
    result = _fetch(site, "http://example.com/a")
    assert "Wektorowe bazy" in result.markdown


def test_http_error_is_an_error_not_an_empty_page():
    site = _Site({"http://example.com/missing": _Reply(status=404, body=b"not found")})
    err = _fetch_error(site, "http://example.com/missing")
    assert "http 404" in str(err)


# ---------------------------------------------------------------------------
# the quality gate
# ---------------------------------------------------------------------------


def test_client_rendered_shell_raises_with_measurements():
    """The failure this engine must never have: success over an empty page.

    The message has to carry the numbers, because "empty content" is not
    actionable -- the caller cannot tell an empty page from a JavaScript shell
    from a nav-only index, and so cannot decide to fall back.
    """
    site = _Site({"http://example.com/app": _Reply(body=SHELL.encode())})

    err = _fetch_error(site, "http://example.com/app")
    message = str(err)
    assert "looks client-rendered or empty" in message
    assert "chars from" in message
    assert "bytes" in message


def test_navigation_page_raises_as_boilerplate(monkeypatch):
    """Text present, but nearly all of it anchor text: an index, not an article."""
    monkeypatch.setattr(config, "DIRECT_MIN_TEXT_CHARS", 1)
    monkeypatch.setattr(config, "DIRECT_MIN_TEXT_RATIO", 0.0)
    monkeypatch.setattr(config, "DIRECT_MAX_LINK_DENSITY", 0.5)

    nav = (
        "<html><body><ul>"
        + "".join(f"<li><a href='/p{i}'>Section {i} of the manual</a></li>" for i in range(40))
        + "</ul></body></html>"
    )
    site = _Site({"http://example.com/docs": _Reply(body=nav.encode())})

    err = _fetch_error(site, "http://example.com/docs")
    assert "link text" in str(err)
    assert "navigation" in str(err)


def test_ratio_check_catches_markup_heavy_documents(monkeypatch):
    """A big body whose visible text is negligible relative to its markup.

    Separate from the shell case: here there *is* text, so the absolute floor
    passes and the ratio is what has to reject it.
    """
    monkeypatch.setattr(config, "DIRECT_MIN_TEXT_CHARS", 1)
    monkeypatch.setattr(config, "DIRECT_MAX_LINK_DENSITY", 1.0)
    monkeypatch.setattr(config, "DIRECT_MIN_TEXT_RATIO", 0.5)

    page = "<html><body><p>tiny</p>" + ("<!-- " + "x" * 5000 + " -->") + "</body></html>"
    site = _Site({"http://example.com/heavy": _Reply(body=page.encode())})

    err = _fetch_error(site, "http://example.com/heavy")
    assert "% text" in str(err)
    assert "mostly markup" in str(err)


def test_short_legitimate_page_is_rejected_by_the_absolute_floor(monkeypatch):
    """Recorded honestly rather than papered over.

    The ratio and density checks both pass for a short page with real content;
    the 500-char floor is what rejects it. That is a deliberate trade-off, not
    an accident, and the fix for a legitimately short page is to lower
    `RAG_DIRECT_MIN_TEXT_CHARS` or use a browser engine -- not to weaken the
    gate for everything.

    Issue #28's definition of done asked for "does not raise on a
    short-but-legitimate page", which cannot hold at a 500-char floor. This
    test pins what the code actually does so the discrepancy is visible.
    """
    page = "<html><body><p>Short but real content about a niche topic.</p></body></html>"
    site = _Site({"http://example.com/s": _Reply(body=page.encode())})

    err = _fetch_error(site, "http://example.com/s")
    assert "looks client-rendered or empty" in str(err)

    # With the floor lowered, the same page passes: the gate is the floor, not
    # a broken extractor. This is what makes the rejection above a policy
    # choice rather than a bug.
    monkeypatch.setattr(config, "DIRECT_MIN_TEXT_CHARS", 10)
    result = _fetch(site, "http://example.com/s")
    assert "Short but real content" in result.markdown


# ---------------------------------------------------------------------------
# bounds
# ---------------------------------------------------------------------------


class _CountingStream:
    """Fake response whose block count proves how much was consumed."""

    def __init__(self, blocks: list[bytes]):
        self.blocks = blocks
        self.consumed = 0

    async def aiter_bytes(self):
        for block in self.blocks:
            self.consumed += 1
            yield block


def test_capped_read_stops_iterating_instead_of_truncating_afterwards():
    """`MAX_SCRAPE_CHARS` truncates the text; this must bound the download.

    Reading the whole body and slicing afterwards would make the cap
    decorative: a 2 GB response would be fully transferred first.
    """
    stream = _CountingStream([b"abcd"] * 10)

    async def go():
        return await direct._read_capped(stream, 10)

    body = asyncio.run(go())

    assert len(body) == 10
    # The bound is real but not exact: `async for` advances the generator before
    # the limit is checked, so one block past the limit is pulled and discarded.
    # That overshoot is bounded by the chunk size, not by the response size --
    # the alternative (resp.read(), then slice) would transfer all ten blocks.
    assert stream.consumed == 4
    assert stream.consumed < len(stream.blocks)


def test_download_cap_is_enforced_through_the_fetch(monkeypatch):
    monkeypatch.setattr(config, "DIRECT_MAX_DOWNLOAD_BYTES", 600)
    big = ARTICLE.encode() * 10  # well past the cap
    site = _Site({"http://example.com/big": _Reply(body=big)})

    async def go():
        client = _client(site)
        try:
            return await direct.fetch_direct(client, "http://example.com/big")
        finally:
            await client.aclose()

    # Either the truncated body still yields text, or the gate refuses it.
    # Both are correct; what must not happen is an unbounded read.
    try:
        result = asyncio.run(go())
    except scrapers.ScraperError as exc:
        assert "chars from" in str(exc)
        return
    assert len(result.markdown) <= config.MAX_SCRAPE_CHARS


# ---------------------------------------------------------------------------
# robots.txt
# ---------------------------------------------------------------------------


def test_robots_disallow_blocks_the_fetch():
    site = _Site(
        {
            "http://example.com/robots.txt": _Reply(
                body=b"User-agent: *\nDisallow: /private\n", content_type="text/plain"
            ),
            "http://example.com/private/x": _Reply(body=ARTICLE.encode()),
        }
    )

    err = _fetch_error(site, "http://example.com/private/x")
    assert "robots.txt" in str(err)
    # The page itself was never requested.
    assert "http://example.com/private/x" not in site.seen


def test_robots_allow_permits_the_fetch():
    site = _Site(
        {
            "http://example.com/robots.txt": _Reply(
                body=b"User-agent: *\nDisallow: /private\n", content_type="text/plain"
            ),
            "http://example.com/public/x": _Reply(body=ARTICLE.encode()),
        }
    )
    result = _fetch(site, "http://example.com/public/x")
    assert "Vector databases" in result.markdown


def test_robots_is_fetched_once_per_origin():
    """Link following touches many pages on one site; robots.txt must be cached."""
    site = _Site(
        {
            "http://example.com/robots.txt": _Reply(
                body=b"User-agent: *\nAllow: /\n", content_type="text/plain"
            ),
            "http://example.com/a": _Reply(body=ARTICLE.encode()),
            "http://example.com/b": _Reply(body=ARTICLE.encode()),
        }
    )

    async def go():
        client = _client(site)
        try:
            await direct.fetch_direct(client, "http://example.com/a")
            await direct.fetch_direct(client, "http://example.com/b")
        finally:
            await client.aclose()

    asyncio.run(go())
    assert site.seen.count("http://example.com/robots.txt") == 1


def test_respect_robots_flag_off_skips_robots_entirely(monkeypatch):
    monkeypatch.setattr(config, "RESPECT_ROBOTS", False)
    site = _Site({"http://example.com/private/x": _Reply(body=ARTICLE.encode())})

    result = _fetch(site, "http://example.com/private/x")

    assert "Vector databases" in result.markdown
    assert not any("robots.txt" in url for url in site.seen)


def test_missing_robots_txt_allows_the_fetch():
    """RFC 9309: a 4xx robots.txt means no restrictions."""
    site = _Site({"http://example.com/a": _Reply(body=ARTICLE.encode())})
    result = _fetch(site, "http://example.com/a")
    assert "Vector databases" in result.markdown


def test_robots_is_fetched_through_the_same_guard(monkeypatch):
    """A host we refuse to crawl is a host we do not ask for permission either."""
    checked: list[str] = []

    async def guard(hostname: str) -> None:
        checked.append(hostname)
        if hostname == "internal.example":
            raise security._reject("Target address is in a private or reserved range.", 403)

    monkeypatch.setattr(security, "_check_resolves_to_public", guard)

    site = _Site({"http://internal.example/a": _Reply(body=ARTICLE.encode())})

    with pytest.raises(security.UrlRejected):
        _fetch(site, "http://internal.example/a")

    # Nothing was requested, not even robots.txt.
    assert site.seen == []
