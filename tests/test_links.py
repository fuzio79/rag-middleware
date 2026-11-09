"""Tests for link resolution and selection during `follow_links`.

These pin the three defects from #12. Each test is written so that it fails
against the previous behaviour rather than merely describing the new one:

* `normalize_url` must not be what gets fetched (it is lossy);
* relative hrefs must resolve against the page that was actually scraped;
* `same_domain` must be evaluated against the *scraped* URL, not the submitted
  one, or a cross-host redirect rejects every link.

Model-free and DB-free.
"""

from __future__ import annotations

import pytest

from rag import api

# ---------------------------------------------------------------------------
# resolve_link
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "link,base,expected",
    [
        ("/docs/page", "https://ex.com/a/b", "https://ex.com/docs/page"),
        ("page2", "https://ex.com/a/b", "https://ex.com/a/page2"),
        ("../up", "https://ex.com/a/b", "https://ex.com/up"),
        ("//cdn.ex.com/x", "https://ex.com/a", "https://cdn.ex.com/x"),
        ("https://other.com/x", "https://ex.com/a", "https://other.com/x"),
        ("?q=2", "https://ex.com/a", "https://ex.com/a?q=2"),
    ],
)
def test_relative_links_are_resolved(link, base, expected):
    assert api.resolve_link(link, base) == expected


@pytest.mark.parametrize(
    "link",
    [
        "javascript:alert(1)",
        "mailto:a@b.c",
        "tel:+123",
        "data:text/html,x",
        "http://[abc",
    ],
)
def test_unsupported_links_return_none(link):
    assert api.resolve_link(link, "https://ex.com/") is None


def test_empty_href_resolves_to_the_page_itself():
    """An empty href means "this page" in HTML, so it must not be followed.

    urljoin(base, "") returns the base, which `select_links` then discards as a
    duplicate of the entry page. Asserting that here documents the behaviour
    rather than leaving it to look like an accident.
    """
    assert api.resolve_link("", "https://ex.com/a") == "https://ex.com/a"
    candidates, skipped = api.select_links(["", "   "], "https://ex.com/a", 10)
    assert candidates == []
    assert skipped == []


# ---------------------------------------------------------------------------
# select_links
# ---------------------------------------------------------------------------


def test_candidates_keep_the_original_url_not_the_normalised_form():
    """The core of the defect: what is fetched must be what was linked.

    `normalize_url` strips a trailing slash, re-quotes `%20` as `+` and drops
    fragments. Fetching the normalised form requests a different resource.
    """
    links = ["https://ex.com/a/", "https://ex.com/b?q=a%20b#frag"]
    candidates, _ = api.select_links(links, base_url="https://ex.com/", limit=10)

    assert candidates == [
        "https://ex.com/a/",
        "https://ex.com/b?q=a%20b#frag",
    ]
    # And prove the assertion is meaningful: normalisation differs for both.
    assert api.normalize_url(candidates[0]) == "https://ex.com/a"
    assert api.normalize_url(candidates[1]) == "https://ex.com/b?q=a+b"


def test_dedupe_still_uses_the_normalised_key():
    """Original URLs are kept for fetching, but duplicates must still collapse."""
    links = ["https://ex.com/a", "https://ex.com/a/", "https://ex.com/a?utm_source=x"]
    candidates, _ = api.select_links(links, base_url="https://ex.com/", limit=10)
    assert candidates == ["https://ex.com/a"]


def test_relative_links_are_followable():
    """Previously every relative href was dropped by is_valid_link."""
    links = ["/docs/one", "docs/two"]
    candidates, skipped = api.select_links(links, base_url="https://ex.com/guide/", limit=10)

    assert candidates == ["https://ex.com/docs/one", "https://ex.com/guide/docs/two"]
    assert skipped == []


def test_off_domain_links_are_excluded():
    links = ["https://other.com/x", "https://ex.com/ok"]
    candidates, _ = api.select_links(links, base_url="https://ex.com/", limit=10)
    assert candidates == ["https://ex.com/ok"]


def test_subdomain_counts_as_same_domain():
    candidates, _ = api.select_links(
        ["https://docs.ex.com/x"], base_url="https://ex.com/", limit=10
    )
    assert candidates == ["https://docs.ex.com/x"]


def test_same_domain_uses_the_scraped_base_not_the_submitted_one():
    """A cross-host entry redirect must not reject every link.

    The page was scraped at `docs.ex.com`, so its links resolve and compare
    against that host. Comparing against the submitted `ex.com` rejected all of
    them and `follow_links` silently did nothing.
    """
    links = ["/a", "/b"]
    scraped_base = "https://docs.ex.com/guide/"

    candidates, _ = api.select_links(links, base_url=scraped_base, limit=10)
    assert candidates == ["https://docs.ex.com/a", "https://docs.ex.com/b"]


def test_the_entry_page_itself_is_excluded():
    links = ["https://ex.com/start", "https://ex.com/next"]
    candidates, _ = api.select_links(links, base_url="https://ex.com/start", limit=10)
    assert candidates == ["https://ex.com/next"]


def test_limit_is_respected():
    links = [f"/p{i}" for i in range(20)]
    candidates, _ = api.select_links(links, base_url="https://ex.com/", limit=3)
    assert len(candidates) == 3


def test_unusable_links_are_reported_not_silently_dropped():
    links = ["javascript:void(0)", "/fine"]
    candidates, skipped = api.select_links(links, base_url="https://ex.com/", limit=10)

    assert candidates == ["https://ex.com/fine"]
    assert skipped == [{"url": "javascript:void(0)", "reason": "not an absolute http(s) link"}]


def test_zero_limit_yields_nothing():
    candidates, _ = api.select_links(["/a"], base_url="https://ex.com/", limit=0)
    assert candidates == []


def test_empty_and_whitespace_links_are_ignored_without_noise():
    candidates, skipped = api.select_links(["", "   "], base_url="https://ex.com/", limit=10)
    assert candidates == []
    assert skipped == []


# ---------------------------------------------------------------------------
# source aliases -- a key change must not orphan a duplicate
# ---------------------------------------------------------------------------


def test_source_aliases_include_prefixed_and_legacy_keys():
    """Every spelling a build may have written must be claimed.

    Since #13 the current build writes `url:<url>`; older builds wrote the raw
    URL and the normalised form. A write has to claim all of them or a
    re-ingest orphans the row under the spelling it did not pick.
    """
    aliases = api.source_aliases("https://ex.com/a", "https://ex.com/a/")
    assert "url:https://ex.com/a" in aliases
    assert "url:https://ex.com/a/" in aliases
    assert "https://ex.com/a" in aliases  # legacy, unprefixed
    assert "https://ex.com/a/" in aliases


def test_source_aliases_are_deduplicated():
    aliases = api.source_aliases("https://ex.com/a")
    assert aliases == list(dict.fromkeys(aliases))
    assert aliases.count("url:https://ex.com/a") == 1


def test_source_aliases_ignore_unparseable_urls():
    """A malformed URL must not make the alias list raise."""
    aliases = api.source_aliases("http://[abc", "https://ex.com/ok")
    assert "url:https://ex.com/ok" in aliases
    assert any("ex.com/ok" in a for a in aliases)


def test_source_aliases_toggle_the_root_slash_both_ways():
    """A bare authority has two spellings and both must be claimed.

    `normalize_url` maps `https://host` and `https://host/` to the same key, so
    an older build could have written either. Claiming only the form that
    happened to be submitted is what left `example.com` sitting under two keys
    in a live database.
    """
    slashed = api.source_aliases("https://example.com/")
    bare = api.source_aliases("https://example.com")

    for form in ("https://example.com", "https://example.com/"):
        assert f"url:{form}" in slashed
        assert f"url:{form}" in bare


def test_the_root_slash_toggle_does_not_apply_to_nested_paths():
    """Only the root path has two spellings; nested paths do not.

    `https://host` and `https://host/` are the same document. `https://host/a/`
    and `https://host/a` are different resources, and conflating them would
    delete data that was never a duplicate.

    `/a` still appears via the *normalised* legacy key -- that is the old key
    scheme being migrated away from, not the slash toggle.
    """
    nested = api.source_aliases("https://ex.com/a/")
    assert "url:https://ex.com/a/" in nested
    assert api.normalize_url("https://ex.com/a/") in nested  # legacy key

    root = api.source_aliases("https://ex.com/")
    root_url_keys = {k for k in root if k.startswith("url:")}
    assert root_url_keys == {"url:https://ex.com/", "url:https://ex.com"}


def test_source_aliases_skip_empty_urls():
    """An empty URL must not become the key "/".

    `normalize_url("")` returns `/`, which no document was ever stored under,
    and deleting by it would match anything else normalising to a bare path.
    """
    assert api.source_aliases("", "") == []
