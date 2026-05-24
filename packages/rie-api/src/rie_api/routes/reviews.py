"""HITL review endpoints — accept/correct/reject decisions, persisted.

Submission is append-only: every review writes a new row so the audit trail
is complete. The list endpoint reads them back ordered by decision time
(repo returns insertion order, which matches `decided_at` in practice).
"""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, status
from rie_contracts import ReviewRecord
from rie_persistence.repository import DocumentRepository

from rie_api.deps import get_repo
from rie_api.errors import NotFoundError, ValidationError
from rie_api.schemas import (
    ReviewDTO,
    ReviewListResponse,
    ReviewSubmit,
)

router = APIRouter(tags=["reviews"])


@router.post(
    "/v1/reviews",
    response_model=ReviewDTO,
    status_code=status.HTTP_201_CREATED,
    summary="Submit a reviewer decision for a claim",
)
def submit_review(
    body: ReviewSubmit,
    repo: DocumentRepository = Depends(get_repo),
) -> ReviewDTO:
    # The claim must exist before we attach a review to it.
    try:
        repo.get_claim(body.claim_id)
    except KeyError as exc:
        raise NotFoundError(f"claim {body.claim_id!r} not found") from exc

    if body.decision.value == "correct" and body.corrected_score is None:
        raise ValidationError("decision=correct requires `corrected_score`")

    record = ReviewRecord(
        claim_id=body.claim_id,
        reviewer=body.reviewer,
        decision=body.decision,
        corrected_score=body.corrected_score,
        note=body.note,
        decided_at=datetime.now(tz=UTC),
    )
    repo.save_review(record)
    return ReviewDTO.from_model(record)


@router.get(
    "/v1/claims/{claim_id}/reviews",
    response_model=ReviewListResponse,
    summary="List all reviews for a claim",
)
def list_reviews_for_claim(
    claim_id: str,
    repo: DocumentRepository = Depends(get_repo),
) -> ReviewListResponse:
    try:
        repo.get_claim(claim_id)
    except KeyError as exc:
        raise NotFoundError(f"claim {claim_id!r} not found") from exc
    reviews = list(repo.list_reviews(claim_id))
    items = [ReviewDTO.from_model(r) for r in reviews]
    return ReviewListResponse(items=items, total=len(items))
