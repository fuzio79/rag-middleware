"""Process-wide app handle.

A tiny module that exists solely to break the import cycle:

    main -> api -> state -> (main)

`api` needs `app.state.scraper` at request time, but importing `main` from `api`
would be circular. Keeping the FastAPI instance on a neutral module lets both
sides reach it.
"""

from __future__ import annotations

from typing import Any


class _State:
    app: Any = None

    @classmethod
    def bind(cls, app: Any) -> None:
        cls.app = app


state = _State()
