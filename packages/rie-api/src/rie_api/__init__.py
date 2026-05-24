"""rie_api — FastAPI driver for the Regulatory Intelligence Engine.

Importing this package is side-effect free (the `app` is constructed in
`main.py`). Public surface stays small: tests and ASGI servers reach for
`rie_api.main:app` or `rie_api.main:create_app`.
"""

from __future__ import annotations

from rie_api.settings import Settings

__all__ = ["Settings"]
