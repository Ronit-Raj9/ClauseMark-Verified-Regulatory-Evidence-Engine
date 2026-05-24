"""Coverage (3-state absence) endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from rie_persistence.repository import DocumentRepository

from rie_api.deps import get_repo
from rie_api.schemas import CoverageDTO, CoverageListResponse

router = APIRouter(prefix="/v1/coverage", tags=["coverage"])


@router.get(
    "",
    response_model=CoverageListResponse,
    summary="List coverage records, optionally filtered by jurisdiction",
)
def list_coverage(
    jurisdiction: str | None = Query(default=None, max_length=64),
    repo: DocumentRepository = Depends(get_repo),
) -> CoverageListResponse:
    records = list(repo.list_coverage(jurisdiction=jurisdiction))
    items = [CoverageDTO.from_model(r) for r in records]
    return CoverageListResponse(items=items, total=len(items))
