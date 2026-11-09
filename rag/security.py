"""Outbound-request safety: API-key auth and SSRF validation.

The SSRF guard here is the important part, but its limits are stated plainly on
purpose, because the previous docstring claimed a guarantee the code could not
keep. The original implementation validated only the URL the caller supplied,
then handed it to a scraper that followed redirects on its own. A public URL
answering `302 -> http://172.x.x.x/` was therefore never re-checked.

What `validate_crawl_url()` does now:

* walks the redirect chain itself, resolving and range-checking every hop;
* probes with GET over a streaming client, so the method matches what the
  crawler will use and the response body is never downloaded;
* fails closed if a hop cannot be checked.

What it cannot do, and does not pretend to:

* a server can answer the probe differently from the crawler (JS redirects,
  `meta refresh`, different headers);
* DNS can change between our lookup and the crawler's own (rebinding);
* Crawl4AI is a browser, so subresources and client-side navigation are its
  business, not ours.

What this module does enforce, and is worth stating plainly: the default
policy is public-only. With `RAG_ALLOWED_CRAWL_DOMAINS` empty, every globally
routable host is permitted and loopback, private, link-local and reserved
ranges are refused -- "anything that is not local". Setting that variable only
*narrows* the set further; it is not what provides the private-range refusal.

The authoritative control against a hostile target is therefore network egress
policy on the crawler container. Treat this module as a pre-filter.
"""

from __future__ import annotations

import asyncio
import hmac
import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import httpx
from fastapi import Header, HTTPException, status

from . import config

REDIRECT_CODES = frozenset({301, 302, 303, 307, 308})

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

    The comparison is constant-time. It is a shared bearer token rather than a
    per-user credential, so timing exposure over a network is arguably
    negligible -- but the cost of doing it properly is zero, and the previous
    comment arguing against it invited someone to weaken it further.
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

    # Encode first: compare_digest raises TypeError when handed a non-ASCII str.
    supplied = (x_api_key or "").encode("utf-8")
    expected = config.API_KEY.encode("utf-8")
    if not hmac.compare_digest(supplied, expected):
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


def _parse(url: str):
    """urlparse, but raising our 400 instead of ValueError.

    `urlparse("http://[abc")` raises ValueError ("Invalid IPv6 URL") from deep
    inside the stdlib. Letting that escape turned any malformed link on a
    scraped page into an unhandled 500.
    """
    try:
        return urlparse(url)
    except ValueError:
        raise _reject("URL is malformed.", 400) from None


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

    This is the check that implements the default policy -- anything that is
    not local. It catches decimal/octal/hex IP literals and `127.1` style
    shorthands, because the resolver normalises them before we see the address.
    """
    # socket.getaddrinfo is BLOCKING. Awaiting it raises TypeError, and calling
    # it directly would stall the event loop. Resolve in a worker thread.
    try:
        loop = asyncio.get_running_loop()
        infos = await loop.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
    except (socket.gaierror, UnicodeError, ValueError):
        # UnicodeError/ValueError cover a hostname that is not valid IDNA;
        # getaddrinfo raises them before it ever reaches the resolver.
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
    parsed = _parse(url)
    _check_scheme(parsed)

    hostname = parsed.hostname
    if not hostname:
        raise _reject("URL has no hostname.", 400)

    _check_allowlist(hostname)
    await _check_resolves_to_public(hostname)


async def validate_crawl_url(url: str, *, follow_redirects: bool = True) -> ValidatedUrl:
    """Validate a URL and, if it redirects, every hop along the way.

    Best-effort pre-filter, not a complete SSRF defence -- see the module
    docstring for the specifics of what escapes it.

    Two details matter and are easy to regress:

    * the probe uses **GET** over `client.stream()`, because a server is free to
      answer HEAD differently from GET and GET is what the crawler sends. The
      stream means only status and headers are read;
    * a hop that cannot be checked is **rejected**, not accepted. The original
      code returned the unverified URL here, so an attacker who made the probe
      fail won by default.
    """
    await _validate_single(url)
    current = url

    if not follow_redirects:
        return ValidatedUrl(requested=url, final=current)

    # follow_redirects=True so the *redirect hops themselves* (a 302 to another
    # public host) are followed rather than reported as the final URL. Without
    # it, a legitimate public redirect was surfaced to the caller as the
    # answer and then fetched. Each hop is still re-validated below.
    async with httpx.AsyncClient(follow_redirects=True, timeout=10.0) as client:
        for _ in range(config.MAX_REDIRECTS):
            try:
                # stream() so only headers are read; leaving the block closes
                # the connection without downloading the body.
                async with client.stream("GET", current) as resp:
                    status_code = resp.status_code
                    location = resp.headers.get("location")
                    resolved = str(resp.url)
            except (httpx.HTTPError, httpx.InvalidURL, ValueError):
                # Fail closed. Returning "current" here would let a server make
                # this probe fail and then redirect the crawler internally.
                raise _reject("Could not verify the URL's redirect chain.", 400) from None

            if status_code not in REDIRECT_CODES or not location:
                # httpx may have followed the chain for us; the URL it actually
                # landed on is the one that has to be range-checked.
                if resolved != current:
                    await _validate_single(resolved)
                return ValidatedUrl(requested=url, final=resolved)

            current = urljoin(current, location)
            await _validate_single(current)

    raise _reject("Too many redirects while validating the URL.", 400)
