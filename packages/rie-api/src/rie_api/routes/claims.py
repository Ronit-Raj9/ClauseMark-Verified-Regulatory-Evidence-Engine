"""Claim listing + detail endpoints.

Claims are the Layer-1 output. `GET /v1/claims/{id}` materialises the full
audit trio: the claim, its latest verification report, and the deterministic
citations (text resolved from `evidence_spans` via the repo — NEVER the LLM).
"""

from __future__ import annotations

from collections.abc import Sequence

from fastapi import APIRouter, Depends, Query
from rie_config.loader import ConfigRepository
from rie_contracts import Claim, EvidenceSpan, Layer1Status
from rie_orchestration.citations import materialise_citations
from rie_orchestration.layer2 import materialise_layer2_for_claim
from rie_persistence.repository import DocumentRepository

from rie_api.deps import get_config_repo, get_repo
from rie_api.errors import NotFoundError
from rie_api.schemas import (
    CitationDTO,
    ClaimDetailResponse,
    ClaimDTO,
    ClaimListResponse,
    Layer2RecommendationDTO,
    VerificationReportDTO,
)

router = APIRouter(prefix="/v1/claims", tags=["claims"])


def _materialise_citations(
    spans: Sequence[EvidenceSpan], repo: DocumentRepository
) -> list[CitationDTO]:
    """Deterministic ID-replacement via ``rie_domain.build_citation``."""
    span_by_id = {s.span_id: s for s in spans}
    built = materialise_citations(
        spans,
        get_element_text=repo.get_element_text,
        get_element=repo.get_element,
        get_document=repo.get_document,
    )
    out: list[CitationDTO] = []
    for citation in built:
        span = span_by_id[citation.span_id]
        out.append(
            CitationDTO(
                span_id=citation.span_id,
                doc_id=citation.doc_id,
                element_id=span.element_id,
                text=citation.snippet,
                char_start=citation.char_start,
                char_end=citation.char_end,
                role=span.role.value,
            )
        )
    return out


def _list_all_claims(
    repo: DocumentRepository,
    jurisdiction: str | None,
    pillar_id: str | None,
    status: Layer1Status | None,
) -> list[Claim]:
    """Combine flagged-queue + any other status via repo helpers.

    The repo's `list_claims_for_review` is the only listing API in the port;
    when `status` is None we widen by querying each status that exists.
    """
    if status is Layer1Status.FLAGGED:
        return list(repo.list_claims_for_review(jurisdiction=jurisdiction, pillar_id=pillar_id))
    # The repo's listing surface is intentionally narrow (review queue only).
    # We fall back to it for FLAGGED; other statuses use a session scan so we
    # don't grow the port for the HTTP layer.
    from rie_persistence.models import ClaimRow
    from rie_persistence.repository import _to_claim  # type: ignore[attr-defined]
    from sqlalchemy import select

    with repo.session_factory() as session:
        stmt = select(ClaimRow)
        if status is not None:
            stmt = stmt.where(ClaimRow.layer1_status == status.value)
        if jurisdiction:
            stmt = stmt.where(ClaimRow.jurisdiction == jurisdiction)
        if pillar_id:
            stmt = stmt.where(ClaimRow.pillar_id == pillar_id)
        rows = session.scalars(stmt).all()
        return [_to_claim(r, list(r.spans)) for r in rows]


@router.get(
    "",
    response_model=ClaimListResponse,
    summary="List claims filtered by jurisdiction / pillar / status",
)
def list_claims(
    jurisdiction: str | None = Query(default=None, max_length=64),
    pillar_id: str | None = Query(default=None, max_length=8),
    status: Layer1Status | None = Query(default=None),
    repo: DocumentRepository = Depends(get_repo),
) -> ClaimListResponse:
    claims = _list_all_claims(repo, jurisdiction, pillar_id, status)
    items = [ClaimDTO.from_model(c) for c in claims]
    return ClaimListResponse(items=items, total=len(items))


@router.get(
    "/{claim_id}",
    response_model=ClaimDetailResponse,
    summary="Full claim + latest verification + citations",
)
def get_claim(
    claim_id: str,
    repo: DocumentRepository = Depends(get_repo),
    config_repo: ConfigRepository = Depends(get_config_repo),
) -> ClaimDetailResponse:
    try:
        claim = repo.get_claim(claim_id)
    except KeyError as exc:
        raise NotFoundError(f"claim {claim_id!r} not found") from exc
    report = repo.latest_verification(claim_id)
    citations = _materialise_citations(claim.evidence_spans, repo)
    layer2_rec = None
    getter = getattr(repo, "get_layer2_recommendation", None)
    if callable(getter):
        try:
            layer2_rec = getter(claim_id)
        except KeyError:
            pass
    if layer2_rec is None:
        coverage_records = list(repo.list_coverage(jurisdiction=claim.jurisdiction))
        layer2_rec = materialise_layer2_for_claim(
            claim,
            coverage_records=coverage_records,
            config=config_repo,
        )
    return ClaimDetailResponse(
        claim=ClaimDTO.from_model(claim),
        verification=VerificationReportDTO.from_model(report) if report else None,
        citations=citations,
        layer2=Layer2RecommendationDTO.from_model(layer2_rec) if layer2_rec else None,
    )
