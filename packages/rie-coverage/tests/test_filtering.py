"""Focused tests for the claim-filtering rules.

Wrong-jurisdiction or wrong-indicator claims must NEVER count as evidence,
and only VERIFIED status counts (PENDING / FLAGGED / REJECTED are skipped).
"""

from __future__ import annotations

import pytest
from rie_contracts import CoverageState, Layer1Status
from rie_coverage import CoverageReasoner
from rie_coverage.policy import filter_qualifying_claims

from .conftest import make_claim


@pytest.fixture
def reasoner() -> CoverageReasoner:
    return CoverageReasoner()


class TestWrongJurisdictionFilteredOut:
    def test_classify_ignores_other_jurisdictions(self, reasoner: CoverageReasoner) -> None:
        claims = [
            make_claim(claim_id="ot", jurisdiction="OTHER"),
            make_claim(claim_id="xx", jurisdiction="XX"),
        ]
        rec = reasoner.evaluate(
            jurisdiction="SAMPLE",
            indicator_id="6.4",
            verified_claims=claims,
            gold_recall=0.9,
        )
        assert rec.state == CoverageState.NO_EVIDENCE_IN_SEARCHED_CORPUS
        assert rec.verified_claim_ids == []

    def test_filter_function_drops_wrong_jurisdictions(self) -> None:
        claims = [
            make_claim(claim_id="ours", jurisdiction="SAMPLE"),
            make_claim(claim_id="other", jurisdiction="OTHER"),
        ]
        kept = filter_qualifying_claims(claims, indicator_id="6.4", jurisdiction="SAMPLE")
        assert [c.claim_id for c in kept] == ["ours"]


class TestWrongIndicatorFilteredOut:
    def test_classify_ignores_other_indicators(self, reasoner: CoverageReasoner) -> None:
        claims = [
            make_claim(claim_id="i71", indicator_id="7.1"),
            make_claim(claim_id="i72", indicator_id="7.2"),
        ]
        rec = reasoner.evaluate(
            jurisdiction="SAMPLE",
            indicator_id="6.4",
            verified_claims=claims,
            gold_recall=0.9,
        )
        assert rec.state == CoverageState.NO_EVIDENCE_IN_SEARCHED_CORPUS
        assert rec.verified_claim_ids == []

    def test_filter_function_drops_wrong_indicators(self) -> None:
        claims = [
            make_claim(claim_id="match", indicator_id="6.4"),
            make_claim(claim_id="nope", indicator_id="7.1"),
        ]
        kept = filter_qualifying_claims(claims, indicator_id="6.4", jurisdiction="SAMPLE")
        assert [c.claim_id for c in kept] == ["match"]


class TestOnlyVerifiedStatusCounts:
    @pytest.mark.parametrize(
        "skipped_status",
        [
            Layer1Status.PENDING_VERIFICATION,
            Layer1Status.FLAGGED,
            Layer1Status.REJECTED,
        ],
    )
    def test_non_verified_statuses_are_skipped(
        self, reasoner: CoverageReasoner, skipped_status: Layer1Status
    ) -> None:
        claims = [make_claim(claim_id="x", layer1_status=skipped_status)]
        rec = reasoner.evaluate(
            jurisdiction="SAMPLE",
            indicator_id="6.4",
            verified_claims=claims,
            gold_recall=0.9,
        )
        assert rec.state == CoverageState.NO_EVIDENCE_IN_SEARCHED_CORPUS
        assert rec.verified_claim_ids == []

    def test_verified_alongside_non_verified_is_kept(self, reasoner: CoverageReasoner) -> None:
        claims = [
            make_claim(claim_id="p", layer1_status=Layer1Status.PENDING_VERIFICATION),
            make_claim(claim_id="v", layer1_status=Layer1Status.VERIFIED),
            make_claim(claim_id="f", layer1_status=Layer1Status.FLAGGED),
            make_claim(claim_id="r", layer1_status=Layer1Status.REJECTED),
        ]
        rec = reasoner.evaluate(
            jurisdiction="SAMPLE",
            indicator_id="6.4",
            verified_claims=claims,
            gold_recall=0.9,
        )
        assert rec.state == CoverageState.EVIDENCE_FOUND
        assert rec.verified_claim_ids == ["v"]


class TestCombinedFiltering:
    """Mixed populations: verified + correct ind + correct jurisdiction wins."""

    def test_only_matching_claim_survives(self, reasoner: CoverageReasoner) -> None:
        claims = [
            # right indicator, right jurisdiction, wrong status
            make_claim(
                claim_id="bad_status",
                layer1_status=Layer1Status.FLAGGED,
                indicator_id="6.4",
                jurisdiction="SAMPLE",
            ),
            # right status, wrong indicator, right jurisdiction
            make_claim(
                claim_id="bad_ind",
                layer1_status=Layer1Status.VERIFIED,
                indicator_id="7.1",
                jurisdiction="SAMPLE",
            ),
            # right status, right indicator, wrong jurisdiction
            make_claim(
                claim_id="bad_juris",
                layer1_status=Layer1Status.VERIFIED,
                indicator_id="6.4",
                jurisdiction="OTHER",
            ),
            # all-right — the only survivor
            make_claim(
                claim_id="winner",
                layer1_status=Layer1Status.VERIFIED,
                indicator_id="6.4",
                jurisdiction="SAMPLE",
            ),
        ]
        rec = reasoner.evaluate(
            jurisdiction="SAMPLE",
            indicator_id="6.4",
            verified_claims=claims,
            gold_recall=0.7,
        )
        assert rec.state == CoverageState.EVIDENCE_FOUND
        assert rec.verified_claim_ids == ["winner"]
