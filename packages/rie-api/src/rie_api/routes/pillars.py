"""Pillar registry endpoints.

The registry is the master index iterated by the orchestrator; the API just
surfaces it (and the per-pillar config) for the UI and external tooling.
NO pillar id is hard-coded here — the data drives the response.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from rie_config.loader import ConfigError, ConfigRepository

from rie_api.deps import get_config_repo
from rie_api.errors import NotFoundError
from rie_api.schemas import (
    PillarDetailResponse,
    PillarsListResponse,
    PillarSummary,
)

router = APIRouter(prefix="/v1/pillars", tags=["pillars"])


@router.get(
    "",
    response_model=PillarsListResponse,
    summary="List all pillars in the registry",
)
def list_pillars(
    config_repo: ConfigRepository = Depends(get_config_repo),
) -> PillarsListResponse:
    entries = list(config_repo.load_registry())
    items = [PillarSummary.from_entry(e) for e in entries]
    return PillarsListResponse(items=items, total=len(items))


@router.get(
    "/{pillar_id}",
    response_model=PillarDetailResponse,
    summary="Full pillar config",
)
def get_pillar(
    pillar_id: str,
    config_repo: ConfigRepository = Depends(get_config_repo),
) -> PillarDetailResponse:
    try:
        pillar = config_repo.load_pillar(pillar_id)
    except ConfigError as exc:
        raise NotFoundError(f"pillar {pillar_id!r}: {exc}") from exc
    return PillarDetailResponse.from_model(pillar)
