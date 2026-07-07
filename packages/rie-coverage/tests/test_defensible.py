"""Tests for CoverageReasoner.evaluate_defensible (Phase 2 §7 upgrade).

Invariants verified:
* schema unchanged — still exactly one of three CoverageState values;
* verified evidence → EVIDENCE_FOUND, unchanged from evaluate();
* no-evidence + high recall + high reach → state STILL
  NO_EVIDENCE_IN_SEARCHED_CORPUS, reason mentions "defensible";
* no-evidence + low reach → reason mentions "bounded"/"unreachable";
* a bare 0 is never written into the record.
"""

from __future__ import annotations

import pytest
from rie_contracts import CoverageRecord, CoverageState
from rie_coverage import CoverageReasoner

from .conftest import make_claim

LEGAL_STATES = {
    CoverageState.EVIDENCE_FOUND,
    CoverageState.NO_EVIDENCE_IN_SEARCHED_CORPUS,
    CoverageState.INSUFFICIENT_COVERAGE,
}


@pytest.fixture
def reasoner() -> CoverageReasoner:
    return CoverageReasoner()


def _assert_no_bare_zero(rec: CoverageRecord) -> None:
    assert rec.state in LEGAL_STATES
    # measured_recall is a recall number, never a "score"; the record carries
    # no score field at all — so there is structurally no 0-score to emit.
    assert not hasattr(rec, "score")


def test_evidence_found_passthrough(reasoner: CoverageReasoner) -> None:
    rec = reasoner.evaluate_defensible(
        jurisdiction="SAMPLE",
        indicator_id="6.4",
        verified_claims=[make_claim(claim_id="c1")],
        gold_recall=0.9,
        reachability=1.0,
    )
    assert rec.state == CoverageState.EVIDENCE_FOUND
    assert rec.verified_claim_ids == ["c1"]
    assert rec.reason is None
    _assert_no_bare_zero(rec)


def test_no_evidence_high_recall_high_reach_is_defensible(
    reasoner: CoverageReasoner,
) -> None:
    rec = reasoner.evaluate_defensible(
        jurisdiction="SAMPLE",
        indicator_id="6.4",
        verified_claims=[],
        gold_recall=0.9,
        reachability=0.95,
    )
    # State must NOT change — still bounded no-evidence, never a 0.
    assert rec.state == CoverageState.NO_EVIDENCE_IN_SEARCHED_CORPUS
    assert rec.measured_recall == 0.9
    assert rec.reason is not None
    assert "defensible" in rec.reason.lower()
    _assert_no_bare_zero(rec)


def test_no_evidence_low_reach_is_bounded(reasoner: CoverageReasoner) -> None:
    rec = reasoner.evaluate_defensible(
        jurisdiction="SAMPLE",
        indicator_id="6.4",
        verified_claims=[],
        gold_recall=0.9,
        reachability=0.3,
    )
    assert rec.state == CoverageState.NO_EVIDENCE_IN_SEARCHED_CORPUS
    assert rec.reason is not None
    low = rec.reason.lower()
    assert "bounded" in low or "unreachable" in low
    assert "defensible-0 candidate" not in low
    _assert_no_bare_zero(rec)


def test_no_evidence_low_recall_is_bounded(reasoner: CoverageReasoner) -> None:
    rec = reasoner.evaluate_defensible(
        jurisdiction="SAMPLE",
        indicator_id="6.4",
        verified_claims=[],
        gold_recall=0.4,
        reachability=1.0,
    )
    assert rec.state == CoverageState.NO_EVIDENCE_IN_SEARCHED_CORPUS
    assert rec.reason is not None
    assert "bounded" in rec.reason.lower()
    _assert_no_bare_zero(rec)


def test_default_reachability_is_conservative(reasoner: CoverageReasoner) -> None:
    # Omitting reachability must never make an absence defensible on its own.
    rec = reasoner.evaluate_defensible(
        jurisdiction="SAMPLE",
        indicator_id="6.4",
        verified_claims=[],
        gold_recall=0.99,
    )
    assert rec.state == CoverageState.NO_EVIDENCE_IN_SEARCHED_CORPUS
    assert rec.reason is not None
    assert "defensible-0 candidate" not in rec.reason.lower()
    _assert_no_bare_zero(rec)


def test_no_recall_is_insufficient_coverage(reasoner: CoverageReasoner) -> None:
    rec = reasoner.evaluate_defensible(
        jurisdiction="SAMPLE",
        indicator_id="6.4",
        verified_claims=[],
        gold_recall=None,
        reachability=1.0,
    )
    assert rec.state == CoverageState.INSUFFICIENT_COVERAGE
    assert rec.measured_recall is None
    _assert_no_bare_zero(rec)


def test_bare_zero_never_emitted_across_inputs(reasoner: CoverageReasoner) -> None:
    inputs = [
        ([], None, None),
        ([], 0.0, 0.0),
        ([], 0.5, 1.0),
        ([], 1.0, 1.0),
        ([], 0.9, 0.95),
        ([make_claim()], None, 1.0),
        ([make_claim()], 0.5, 0.0),
    ]
    for claims, recall, reach in inputs:
        rec = reasoner.evaluate_defensible(
            jurisdiction="SAMPLE",
            indicator_id="6.4",
            verified_claims=claims,
            gold_recall=recall,
            reachability=reach,
        )
        _assert_no_bare_zero(rec)


def test_original_evaluate_unchanged(reasoner: CoverageReasoner) -> None:
    # Back-compat: evaluate() must not gain defensibility text.
    rec = reasoner.evaluate(
        jurisdiction="SAMPLE",
        indicator_id="6.4",
        verified_claims=[],
        gold_recall=0.9,
    )
    assert rec.state == CoverageState.NO_EVIDENCE_IN_SEARCHED_CORPUS
    assert rec.reason is not None
    assert "defensible" not in rec.reason.lower()
