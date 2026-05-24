"""Liveness + readiness probes.

`/healthz` is a cheap "process is up" probe (no I/O).
`/readyz` probes downstream dependencies (Postgres + Qdrant) so a load
balancer can pull traffic when the system can't actually serve a request.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from fastapi import APIRouter, Depends
from rie_persistence.repository import DocumentRepository
from sqlalchemy import text

from rie_api.deps import get_repo, get_settings
from rie_api.schemas import HealthResponse
from rie_api.settings import Settings

logger = logging.getLogger("rie_api.health")

router = APIRouter(tags=["health"])


@router.get("/healthz", response_model=HealthResponse, summary="Liveness probe")
async def healthz() -> HealthResponse:
    """No I/O — confirms the process is responsive."""
    return HealthResponse(status="ok", components={"api": "ok"})


@router.get("/readyz", response_model=HealthResponse, summary="Readiness probe")
async def readyz(
    settings: Settings = Depends(get_settings),
    repo: DocumentRepository = Depends(get_repo),
) -> HealthResponse:
    """Probe Postgres + Qdrant. 200 with degraded components when partial.

    We never raise here — the response shape itself carries the per-component
    state so dashboards can surface "Postgres up, Qdrant down".
    """
    components: dict[str, str] = {"api": "ok"}
    overall_ok = True

    # ── Postgres ──
    try:
        with repo.session_factory() as session:
            session.execute(text("SELECT 1"))
        components["postgres"] = "ok"
    except Exception as exc:
        logger.warning("postgres probe failed: %s", exc)
        components["postgres"] = f"error: {type(exc).__name__}"
        overall_ok = False

    # ── Qdrant ──
    qdrant_url = f"{settings.qdrant_base_url()}/healthz"
    try:
        async with httpx.AsyncClient(timeout=1.5) as client:
            resp = await client.get(qdrant_url)
        components["qdrant"] = "ok" if resp.status_code == 200 else f"http_{resp.status_code}"
        if resp.status_code != 200:
            overall_ok = False
    except Exception as exc:
        logger.warning("qdrant probe failed: %s", exc)
        components["qdrant"] = f"error: {type(exc).__name__}"
        overall_ok = False

    return HealthResponse(
        status="ok" if overall_ok else "degraded",
        components=components,
    )


# Help static analysers that flag the `Any` import in some configs.
_ANY: Any = None
