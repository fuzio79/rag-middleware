"""Tests for the SSRF guard and API-key dependency.

These do not need a LanceDB or an embedding model; they cover the logic that
decides whether an outbound request is allowed to happen at all.

Run: pytest -q
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException

from rag import config, security

# ---------------------------------------------------------------------------
# auth
# ---------------------------------------------------------------------------


def _set_auth(monkeypatch, *, key: str, required: bool):
    monkeypatch.setattr(config, "API_KEY", key)
    monkeypatch.setattr(config, "REQUIRE_AUTH", required)


def _call(key):
    return asyncio.run(security.require_api_key(key))


def test_correct_key_allowed(monkeypatch):
    _set_auth(monkeypatch, key="s3cret", required=True)
    assert _call("s3cret") is None


def test_wrong_key_rejected(monkeypatch):
    _set_auth(monkeypatch, key="s3cret", required=True)
    with pytest.raises(HTTPException) as exc:
        _call("wrong")
    assert exc.value.status_code == 401


def test_missing_key_rejected(monkeypatch):
    _set_auth(monkeypatch, key="s3cret", required=True)
    with pytest.raises(HTTPException) as exc:
        _call(None)
    assert exc.value.status_code == 401


def test_unset_server_key_fails_closed(monkeypatch):
    """Auth required but no key configured must refuse, not serve openly."""
    _set_auth(monkeypatch, key="", required=True)
    with pytest.raises(HTTPException) as exc:
        _call("anything")
    assert exc.value.status_code == 503


def test_auth_disabled_allows_everything(monkeypatch):
    _set_auth(monkeypatch, key="", required=False)
    assert _call(None) is None


# ---------------------------------------------------------------------------
# SSRF: direct targets
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8000/health",
        "http://localhost:8000/",
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata
        "http://172.23.0.5:8000/",  # docker bridge
        "http://[::1]:8000/",
        "http://2130706433:8000/",  # decimal 127.0.0.1
        "http://0x7f000001:8000/",  # hex 127.0.0.1
        "http://127.1:8000/",  # shorthand
        "http://0177.0.0.1:8000/",  # octal
    ],
)
def test_private_targets_rejected(url, monkeypatch):
    monkeypatch.setattr(config, "ALLOWED_CRAWL_DOMAINS", frozenset())
    with pytest.raises(HTTPException) as exc:
        asyncio.run(security.validate_crawl_url(url, follow_redirects=False))
    assert exc.value.status_code == 403


@pytest.mark.parametrize(
    "url", ["file:///etc/passwd", "ftp://example.com/x", "javascript:alert(1)"]
)
def test_non_http_schemes_rejected(url, monkeypatch):
    monkeypatch.setattr(config, "ALLOWED_CRAWL_DOMAINS", frozenset())
    with pytest.raises(HTTPException) as exc:
        asyncio.run(security.validate_crawl_url(url, follow_redirects=False))
    assert exc.value.status_code == 400


def test_unresolvable_host_rejected(monkeypatch):
    """DNS failure must be a 400, not a silent pass.

    Patched rather than relying on a real NXDOMAIN: some container resolvers
    answer for `.invalid` with a private address, which would make this test
    assert on the environment instead of on our own behaviour.
    """
    import socket

    async def boom(*args, **kwargs):
        raise socket.gaierror("NXDOMAIN")

    class _Loop:
        async def getaddrinfo(self, *a, **k):
            raise socket.gaierror("NXDOMAIN")

    monkeypatch.setattr(security.asyncio, "get_running_loop", lambda: _Loop())
    monkeypatch.setattr(config, "ALLOWED_CRAWL_DOMAINS", frozenset())

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            security.validate_crawl_url(
                "http://this-host-does-not-exist.invalid/", follow_redirects=False
            )
        )
    assert exc.value.status_code == 400


# ---------------------------------------------------------------------------
# SSRF: allowlist
# ---------------------------------------------------------------------------


def test_allowlist_blocks_other_domains(monkeypatch):
    monkeypatch.setattr(config, "ALLOWED_CRAWL_DOMAINS", frozenset({"allowed.example"}))
    with pytest.raises(HTTPException) as exc:
        asyncio.run(security.validate_crawl_url("http://other.example/", follow_redirects=False))
    assert exc.value.status_code == 403


def test_allowlist_permits_subdomain(monkeypatch):
    """A subdomain of an allowed domain passes the allowlist check itself."""
    monkeypatch.setattr(config, "ALLOWED_CRAWL_DOMAINS", frozenset({"example.com"}))
    # Should get past the allowlist and fail later only on DNS, proving the
    # allowlist accepted it.
    assert security._check_allowlist("docs.example.com") is None


def test_allowlist_rejects_lookalike(monkeypatch):
    monkeypatch.setattr(config, "ALLOWED_CRAWL_DOMAINS", frozenset({"example.com"}))
    with pytest.raises(HTTPException):
        security._check_allowlist("notexample.com")


# ---------------------------------------------------------------------------
# SSRF: redirect re-validation (the Phase 2 fix)
# ---------------------------------------------------------------------------


class _Resp:
    def __init__(self, status_code, location=None):
        self.status_code = status_code
        self.headers = {} if location is None else {"location": location}


class _Client:
    """Fake httpx client returning a scripted redirect chain."""

    script: dict[str, _Resp] = {}

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def head(self, url):
        return self.script.get(url, _Resp(200))


@pytest.fixture
def redirect_env(monkeypatch):
    """Patch out DNS so we control which hops look private."""
    seen: list[str] = []

    async def fake_check(hostname: str):
        seen.append(hostname)
        if hostname in {"127.0.0.1", "169.254.169.254", "10.0.0.1"}:
            raise security._reject("Target address is in a private or reserved range.", 403)

    monkeypatch.setattr(security, "_check_resolves_to_public", fake_check)
    monkeypatch.setattr(security.httpx, "AsyncClient", _Client)
    monkeypatch.setattr(config, "ALLOWED_CRAWL_DOMAINS", frozenset())
    return seen


def test_redirect_to_private_is_blocked(redirect_env):
    """The exact hole the original code had."""
    _Client.script = {"http://public.example/evil": _Resp(302, "http://127.0.0.1:8000/")}
    with pytest.raises(HTTPException) as exc:
        asyncio.run(security.validate_crawl_url("http://public.example/evil"))
    assert exc.value.status_code == 403
    # Both the entry point and the redirect target were resolved.
    assert "public.example" in redirect_env
    assert "127.0.0.1" in redirect_env


def test_redirect_to_metadata_endpoint_blocked(redirect_env):
    _Client.script = {
        "http://public.example/x": _Resp(302, "http://169.254.169.254/latest/meta-data/")
    }
    with pytest.raises(HTTPException) as exc:
        asyncio.run(security.validate_crawl_url("http://public.example/x"))
    assert exc.value.status_code == 403


def test_public_redirect_is_followed_and_allowed(redirect_env):
    _Client.script = {"http://a.example/": _Resp(302, "http://b.example/")}
    result = asyncio.run(security.validate_crawl_url("http://a.example/"))
    assert result.final == "http://b.example/"


def test_redirect_loop_is_bounded(redirect_env, monkeypatch):
    monkeypatch.setattr(config, "MAX_REDIRECTS", 3)
    _Client.script = {"http://loop.example/": _Resp(302, "http://loop.example/")}
    with pytest.raises(HTTPException) as exc:
        asyncio.run(security.validate_crawl_url("http://loop.example/"))
    assert exc.value.status_code == 400
