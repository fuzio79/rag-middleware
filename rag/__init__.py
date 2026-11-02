"""rag-middleware package.

Import surface is deliberately small: `create_app()` is the only thing callers
(e.g. uvicorn's `main:app`) need.
"""

from __future__ import annotations

__all__ = ["create_app"]


def create_app(*args, **kwargs):
    """Lazy re-export so importing the package does not pull in FastAPI."""
    from .app import create_app as _create_app

    return _create_app(*args, **kwargs)
