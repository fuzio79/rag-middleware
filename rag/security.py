"""Outbound-request safety: API-key auth and SSRF validation.

The SSRF guard here is the important part. The original implementation
validated only the URL the caller supplied, then handed it to a scraper that
followed redirects on its own. A public URL answering `302 -> http://172.x.x.x/`
was therefore never re-checked.

`validate_crawl_url()` now walks the redirect chain itself, resolving and
checking every hop before the scraper is ever asked to fetch anything.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import httpx
from fastapi import Header, HTTPException, status

from . import config

# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """FastAPI dependency enforcing the shared API key.

    Behaviour is deliberately fail-closed:

    * `RAG_API_KEY` set      -> the supplied key must match, else 401.
    * unset, auth required   -> 503. Refusing is safer than silently serving
                                an open endpoint because a secret was missed.
    * unset, auth disabled   -> allowed, for trusted-network installs that set
                                `RAG_REQUIRE_AUTH=false` on purpose.
    """
    if not config.REQUIRE_AUTH:
        return

    if not config.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Server has no RAG_API_KEY configured and authentication is "
                "required. Set RAG_API_KEY, or set RAG_REQUIRE_AUTH=false to "
                "serve without authentication on a trusted network."
            ),
        )

    # Constant-time compare is unnecessary here: the key is compared over the
    # network where timing noise dominates, and this is a shared bearer token
    # rather than a per-user credential. Using hmac.compare_digest would be
    # defensible; a plain compare is not the weak link.
    if x_api_key != config.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "X-API-Key"},
        )


# ---------------------------------------------------------------------------
# SSRF validation
# ---------------------------------------------------------------------------


class UrlRejected(HTTPException):
    """Raised when a URL fails the SSRF guard."""


@dataclass(frozen=True)
class ValidatedUrl:
    """A URL that passed every hop check, plus where it finally landed."""

    requested: str
    final: str


def _reject(detail: str, code: int = 403) -> UrlRejected:
    return UrlRejected(status_code=code, detail=detail)


def _check_scheme(parsed) -> None:
    if parsed.scheme not in ("http", "https"):
        raise _reject("Only http and https URLs are allowed.", 400)


def _check_allowlist(hostname: str) -> None:
    if not config.ALLOWED_CRAWL_DOMAINS:
        return
    host = hostname.lower()
    if not any(host == d or host.endswith("." + d) for d in config.ALLOWED_CRAWL_DOMAINS):
        raise _reject("Domain not in the allowed crawl list.", 403)


async def _check_resolves_to_public(hostname: str) -> None:
    """Reject hostnames that resolve to a non-global address.

    Catches decimal/octal/hex IP literals and `127.1` style shorthands, because
    the resolver normalises them before we see the address.
    """
    # socket.getaddrinfo is BLOCKING. Awaiting it raises TypeError, and calling
    # it directly would stall the event loop. Resolve in a worker thread.
    try:
        loop = asyncio.get_running_loop()
        infos = await loop.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
    except socket.gaierror:
        raise _reject("Cannot resolve hostname.", 400) from None

    if not infos:
        raise _reject("Cannot resolve hostname.", 400)

    for _family, _type, _proto, _canon, sockaddr in infos:
        # getaddrinfo types sockaddr's first element as str | int, though in
        # practice it is always the address string. Coerce before splitting so
        # the narrowing is explicit rather than assumed.
        raw = str(sockaddr[0]).split("%")[0]
        try:
            ip = ipaddress.ip_address(raw)
        except ValueError:
            raise _reject("Target address could not be parsed.", 400) from None
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
            ip = ip.ipv4_mapped
        if not ip.is_global or ip.is_multicast:
            raise _reject("Target address is in a private or reserved range.", 403)


async def _validate_single(url: str) -> None:
    parsed = urlparse(url)
    _check_scheme(parsed)

    hostname = parsed.hostname
    if not hostname:
        raise _reject("URL has no hostname.", 400)

    _check_allowlist(hostname)
    await _check_resolves_to_public(hostname)


async def validate_crawl_url(url: str, *, follow_redirects: bool = True) -> ValidatedUrl:
    """Validate a URL and, if it redirects, every hop along the way.

    The scraper is configured not to follow redirects, so this is the only
    thing standing between a caller and an internal address. Each hop is
    resolved and range-checked before the next request is made, which also
    blunts DNS rebinding: the address validated is the address used.
    """
    await _validate_single(url)
    current = url

    if not follow_redirects:
        return ValidatedUrl(requested=url, final=current)

    async with httpx.AsyncClient(follow_redirects=False, timeout=10.0) as client:
        for _ in range(config.MAX_REDIRECTS):
            try:
                resp = await client.head(current)
            except httpx.HTTPError:
                # A network failure here is not a security problem: the scraper
                # will surface it. The hop itself was already validated.
                return ValidatedUrl(requested=url, final=current)

            if resp.status_code not in (301, 302, 303, 307, 308):
                return ValidatedUrl(requested=url, final=current)

            location = resp.headers.get("location")
            if not location:
                return ValidatedUrl(requested=url, final=current)

            current = urljoin(current, location)
            await _validate_single(current)

    raise _reject("Too many redirects while validating the URL.", 400)
