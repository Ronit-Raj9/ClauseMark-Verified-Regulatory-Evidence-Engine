"""review_node behaviour — skipped when no FLAGGED, interrupts when present."""

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
    VerificationReport,
    VerificationStatus,
)
from rie_orchestration.fakes import FakeRepo
from rie_orchestration.review_node import review_node
from rie_orchestration.state import RieState
from rie_orchestration.wiring import AdapterBundle


def _t() -> datetime:
    return datetime(2026, 5, 24, tzinfo=UTC)


def _bundle(repo: FakeRepo) -> AdapterBundle:
    from rie_orchestration.fakes import (
        FakeClassifier,
        FakeCoverage,
        FakeExtractor,
        FakeIngest,
        FakeRetrieval,
        FakeVerifier,
    )

    return AdapterBundle(
        config=None,  # type: ignore[arg-type]
        ingest=FakeIngest(None),  # type: ignore[arg-type]
        extractor=FakeExtractor(),
        retrieval=FakeRetrieval(),
        classifier=FakeClassifier(get_element=repo.get_element),
        verifier=FakeVerifier(),
        coverage=FakeCoverage(),
        repo=repo,
        samples_dir=__import__("pathlib").Path("."),
    )


def _claim(cid: str) -> Claim:
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
        layer1_status=Layer1Status.FLAGGED,
        created_at=_t(),
    )


def _report(cid: str, status: VerificationStatus) -> VerificationReport:
    return VerificationReport(
        claim_id=cid,
        gates=[GateResult(gate=g, passed=True, ran_at=_t()) for g in GateName],
        status=status,
    )


def test_review_node_skips_when_no_flagged() -> None:
    repo = FakeRepo()
    state: RieState = {
        "run_id": "r",
        "jurisdiction": "SAMPLE",
        "pillar_ids": ["6"],
        "claims": [_claim("c1")],
        "verifications": {"c1": _report("c1", VerificationStatus.VERIFIED)},
    }
    out = review_node(state, _bundle(repo))
    assert out["verifications"]["c1"].status == VerificationStatus.VERIFIED


def test_review_node_skips_when_skip_hitl_true() -> None:
    repo = FakeRepo()
    state: RieState = {
        "run_id": "r",
        "jurisdiction": "SAMPLE",
        "pillar_ids": ["6"],
        "skip_hitl": True,
        "claims": [_claim("c1")],
        "verifications": {"c1": _report("c1", VerificationStatus.FLAGGED)},
    }
    out = review_node(state, _bundle(repo))
    assert out["claims"][0].layer1_status == Layer1Status.FLAGGED
