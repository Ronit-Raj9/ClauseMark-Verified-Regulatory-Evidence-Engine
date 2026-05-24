"""End-to-end tests for the :class:`CoverageReasoner` service."""

from __future__ import annotations

import pytest
from rie_contracts import CoverageReasonerPort, CoverageRecord, CoverageState, Layer1Status
from rie_coverage import CoverageReasoner

from .conftest import make_claim


@pytest.fixture
def reasoner() -> CoverageReasoner:
    return CoverageReasoner()


class TestEvaluateEvidenceFound:
    def test_one_verified_claim_yields_evidence_found(self, reasoner: CoverageReasoner) -> None:
        claims = [make_claim(claim_id="c1")]
        rec = reasoner.evaluate(
            jurisdiction="SAMPLE",
            indicator_id="6.4",
            verified_claims=claims,
            gold_recall=0.82,
        )
        assert isinstance(rec, CoverageRecord)
        assert rec.state == CoverageState.EVIDENCE_FOUND
        assert rec.verified_claim_ids == ["c1"]
        assert rec.reason is None
        # measured_recall is passed through even when evidence was found —
        # downstream review still benefits from knowing the corpus recall.
        assert rec.measured_recall == 0.82

    def test_evidence_found_without_recall_pass_through_none(
        self, reasoner: CoverageReasoner
    ) -> None:
        claims = [make_claim(claim_id="c1")]
        rec = reasoner.evaluate(
            jurisdiction="SAMPLE",
            indicator_id="6.4",
            verified_claims=claims,
            gold_recall=None,
        )
        assert rec.state == CoverageState.EVIDENCE_FOUND
        assert rec.verified_claim_ids == ["c1"]
        assert rec.measured_recall is None

    def test_multiple_verified_claims_all_kept(self, reasoner: CoverageReasoner) -> None:
        claims = [
            make_claim(claim_id="c1"),
            make_claim(claim_id="c2"),
            make_claim(claim_id="c3"),
        ]
        rec = reasoner.evaluate(
            jurisdiction="SAMPLE",
            indicator_id="6.4",
            verified_claims=claims,
            gold_recall=0.9,
        )
        assert rec.state == CoverageState.EVIDENCE_FOUND
        assert rec.verified_claim_ids == ["c1", "c2", "c3"]


class TestEvaluateNoEvidence:
    def test_no_claims_with_recall_082(self, reasoner: CoverageReasoner) -> None:
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
        # Per spec: "~{miss_pct}% miss risk based on gold-set recall".
        assert "18%" in rec.reason
        assert "miss risk" in rec.reason

    def test_only_flagged_or_rejected_claims_still_no_evidence(
        self, reasoner: CoverageReasoner
    ) -> None:
        claims = [
            make_claim(claim_id="f1", layer1_status=Layer1Status.FLAGGED),
            make_claim(claim_id="r1", layer1_status=Layer1Status.REJECTED),
            make_claim(claim_id="p1", layer1_status=Layer1Status.PENDING_VERIFICATION),
        ]
        rec = reasoner.evaluate(
            jurisdiction="SAMPLE",
            indicator_id="6.4",
            verified_claims=claims,
            gold_recall=0.5,
        )
        assert rec.state == CoverageState.NO_EVIDENCE_IN_SEARCHED_CORPUS
        assert rec.measured_recall == 0.5
        assert rec.verified_claim_ids == []
        assert rec.reason is not None and "50%" in rec.reason


class TestEvaluateInsufficientCoverage:
    def test_no_claims_no_recall(self, reasoner: CoverageReasoner) -> None:
        rec = reasoner.evaluate(
            jurisdiction="SAMPLE",
            indicator_id="6.4",
            verified_claims=[],
            gold_recall=None,
        )
        assert rec.state == CoverageState.INSUFFICIENT_COVERAGE
        assert rec.measured_recall is None
        assert rec.verified_claim_ids == []
        assert rec.reason == "no gold-set recall measured for this indicator"

    def test_only_flagged_claims_no_recall(self, reasoner: CoverageReasoner) -> None:
        claims = [make_claim(claim_id="f1", layer1_status=Layer1Status.FLAGGED)]
        rec = reasoner.evaluate(
            jurisdiction="SAMPLE",
            indicator_id="6.4",
            verified_claims=claims,
            gold_recall=None,
        )
        assert rec.state == CoverageState.INSUFFICIENT_COVERAGE
        assert rec.measured_recall is None


class TestPortConformance:
    def test_runtime_isinstance(self, reasoner: CoverageReasoner) -> None:
        assert isinstance(reasoner, CoverageReasonerPort)

    def test_bare_zero_never_emitted(self, reasoner: CoverageReasoner) -> None:
        # Sweep the meaningful input space — every outcome lands in one of
        # the three legal states. No silent zero.
        for claims in ([], [make_claim()], [make_claim(layer1_status=Layer1Status.FLAGGED)]):
            for recall in (None, 0.0, 0.5, 1.0):
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
