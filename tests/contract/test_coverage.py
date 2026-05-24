"""Contract test for :class:`rie_contracts.ports.CoverageReasonerPort`.

Any adapter that claims to implement the port must pass this file unmodified.
We import the concrete ``rie_coverage.CoverageReasoner`` and verify:

* runtime ``isinstance`` against the ``@runtime_checkable`` Protocol;
* every one of the three legal :class:`CoverageState` values can be produced;
* a bare ``0`` is never emitted.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from rie_contracts import (
    Claim,
    ClausePattern,
    CoverageReasonerPort,
    CoverageRecord,
    CoverageState,
    Decomposition,
    EvidenceSpan,
    Layer1Status,
    LegalRegime,
)
from rie_coverage import CoverageReasoner

# ────────────────────────────────────────────────────────────────────────────
# Local builders — independent of the package's test fixtures so this file
# stays a true black-box contract test.
# ────────────────────────────────────────────────────────────────────────────


def _verified_claim(
    *,
    claim_id: str = "claim_contract_001",
    indicator_id: str = "6.4",
    jurisdiction: str = "SAMPLE",
    layer1_status: Layer1Status = Layer1Status.VERIFIED,
) -> Claim:
    doc_id = "sample_dpa_2020"
    elem_id = "sample_dpa_2020_s26_p1"
    char_start, char_end = 4120, 4215
    span = EvidenceSpan(
        span_id=f"{doc_id}#{char_start}-{char_end}",
        element_id=elem_id,
        doc_id=doc_id,
        char_start=char_start,
        char_end=char_end,
    )
    return Claim(
        claim_id=claim_id,
        indicator_id=indicator_id,
        pillar_id="6",
        clause_id=elem_id,
        jurisdiction=jurisdiction,
        clause_pattern=ClausePattern.CONDITIONAL_REGIME,
        decomposition=Decomposition(
            subject="organisation",
            condition="transfer outside the country",
            constraint="recipient bound by comparable-protection obligations",
        ),
        evidence_spans=[span],
        regime=LegalRegime(primary_element_id=elem_id, member_element_ids=[elem_id]),
        layer1_status=layer1_status,
        created_at=datetime(2026, 5, 24, tzinfo=UTC),
    )


# ────────────────────────────────────────────────────────────────────────────
# Tests
# ────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def reasoner() -> CoverageReasoner:
    return CoverageReasoner()


def test_runtime_isinstance(reasoner: CoverageReasoner) -> None:
    """The implementation must structurally satisfy the port."""
    assert isinstance(reasoner, CoverageReasonerPort)


def test_evaluate_returns_coverage_record(reasoner: CoverageReasoner) -> None:
    rec = reasoner.evaluate(
        jurisdiction="SAMPLE",
        indicator_id="6.4",
        verified_claims=[_verified_claim()],
        gold_recall=0.82,
    )
    assert isinstance(rec, CoverageRecord)


# ── three legal states, one test each ─────────────────────────────────────


def test_evidence_found(reasoner: CoverageReasoner) -> None:
    rec = reasoner.evaluate(
        jurisdiction="SAMPLE",
        indicator_id="6.4",
        verified_claims=[_verified_claim(claim_id="c1")],
        gold_recall=0.82,
    )
    assert rec.state == CoverageState.EVIDENCE_FOUND
    assert rec.verified_claim_ids == ["c1"]
    assert rec.reason is None


def test_no_evidence_in_searched_corpus_carries_recall(
    reasoner: CoverageReasoner,
) -> None:
    rec = reasoner.evaluate(
        jurisdiction="SAMPLE",
        indicator_id="6.4",
        verified_claims=[],
        gold_recall=0.82,
    )
    assert rec.state == CoverageState.NO_EVIDENCE_IN_SEARCHED_CORPUS
    assert rec.measured_recall == 0.82
    assert rec.verified_claim_ids == []
    assert rec.reason is not None
    # Recall 0.82 → 18% miss risk, per the documented rounding rule.
    assert "18%" in rec.reason


def test_insufficient_coverage_when_no_recall(reasoner: CoverageReasoner) -> None:
    rec = reasoner.evaluate(
        jurisdiction="SAMPLE",
        indicator_id="6.4",
        verified_claims=[],
        gold_recall=None,
    )
    assert rec.state == CoverageState.INSUFFICIENT_COVERAGE
    assert rec.measured_recall is None
    assert rec.verified_claim_ids == []
    assert rec.reason is not None


# ── invariants ─────────────────────────────────────────────────────────────


def test_bare_zero_is_never_emitted(reasoner: CoverageReasoner) -> None:
    """Every observable output of evaluate() is one of three legal states."""
    inputs = [
        ([], None),
        ([], 0.0),
        ([], 0.5),
        ([], 1.0),
        ([_verified_claim()], None),
        ([_verified_claim()], 0.5),
        ([_verified_claim(layer1_status=Layer1Status.FLAGGED)], 0.5),
        ([_verified_claim(layer1_status=Layer1Status.PENDING_VERIFICATION)], None),
    ]
    for claims, recall in inputs:
        rec = reasoner.evaluate(
            jurisdiction="SAMPLE",
            indicator_id="6.4",
            verified_claims=claims,
            gold_recall=recall,
        )
        assert rec.state in {
            CoverageState.EVIDENCE_FOUND,
            CoverageState.NO_EVIDENCE_IN_SEARCHED_CORPUS,
            CoverageState.INSUFFICIENT_COVERAGE,
        }


def test_filtering_jurisdiction_and_indicator(reasoner: CoverageReasoner) -> None:
    """Wrong-jurisdiction or wrong-indicator claims do not count as evidence."""
    claims = [
        _verified_claim(claim_id="wrong_juris", jurisdiction="OTHER"),
        _verified_claim(claim_id="wrong_ind", indicator_id="7.1"),
    ]
    rec = reasoner.evaluate(
        jurisdiction="SAMPLE",
        indicator_id="6.4",
        verified_claims=claims,
        gold_recall=0.9,
    )
    assert rec.state == CoverageState.NO_EVIDENCE_IN_SEARCHED_CORPUS
    assert rec.verified_claim_ids == []
