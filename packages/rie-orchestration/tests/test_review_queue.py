"""HumanReviewQueue behaviour over a FakeRepo."""

from __future__ import annotations

from datetime import UTC, datetime

from rie_contracts import (
    Claim,
    ClausePattern,
    Decomposition,
    EvidenceSpan,
    GateName,
    GateResult,
    Layer1Status,
    LegalRegime,
    ReviewDecision,
    ReviewRecord,
    VerificationReport,
    VerificationStatus,
)
from rie_orchestration.fakes import FakeRepo
from rie_orchestration.review_queue import HumanReviewQueue


def _t() -> datetime:
    return datetime(2026, 5, 24, tzinfo=UTC)


def _claim(cid: str, status: Layer1Status = Layer1Status.FLAGGED) -> Claim:
    return Claim(
        claim_id=cid,
        indicator_id="6.4",
        pillar_id="6",
        clause_id="e1",
        jurisdiction="SAMPLE",
        clause_pattern=ClausePattern.CONDITIONAL_REGIME,
        decomposition=Decomposition(subject="x", constraint="y"),
        evidence_spans=[
            EvidenceSpan(span_id="d#0-1", element_id="e1", doc_id="d", char_start=0, char_end=1)
        ],
        regime=LegalRegime(primary_element_id="e1", member_element_ids=["e1"]),
        layer1_status=status,
        created_at=_t(),
    )


def _report(
    cid: str, status: VerificationStatus = VerificationStatus.FLAGGED
) -> VerificationReport:
    return VerificationReport(
        claim_id=cid,
        gates=[GateResult(gate=g, passed=g != GateName.ENTAILMENT, ran_at=_t()) for g in GateName],
        status=status,
    )


def test_enqueue_returns_claim_id() -> None:
    repo = FakeRepo()
    q = HumanReviewQueue(repo=repo)
    out = q.enqueue(_claim("c1"), _report("c1"))
    assert out == "c1"


def test_resolve_persists_review() -> None:
    repo = FakeRepo()
    q = HumanReviewQueue(repo=repo)
    review = ReviewRecord(
        claim_id="c1",
        reviewer="lead",
        decision=ReviewDecision.ACCEPT,
        decided_at=_t(),
    )
    q.resolve("c1", review)
    assert len(repo.reviews) == 1
    assert repo.reviews[0].claim_id == "c1"
