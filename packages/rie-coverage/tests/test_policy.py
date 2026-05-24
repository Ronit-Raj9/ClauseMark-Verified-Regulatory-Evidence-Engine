"""Exhaustive truth-table tests for the 3-state coverage policy."""

from __future__ import annotations

import pytest
from rie_contracts import CoverageState, Layer1Status
from rie_coverage.policy import (
    classify,
    filter_qualifying_claims,
    format_insufficient_coverage_reason,
    format_no_evidence_reason,
    miss_pct_from_recall,
    requires_authoritative_source_check,
)

from .conftest import make_claim  # local conftest, importlib mode

# ────────────────────────────────────────────────────────────────────────────
# classify() — exhaustive truth table over the 3 states
# ────────────────────────────────────────────────────────────────────────────


class TestClassifyTruthTable:
    """Every cell in the (has_verified_claim, has_gold_recall) matrix."""

    def test_verified_claim_present_with_recall_yields_evidence_found(self) -> None:
        claims = [make_claim()]
        state = classify(claims, gold_recall=0.82, indicator_id="6.4", jurisdiction="SAMPLE")
        assert state == CoverageState.EVIDENCE_FOUND

    def test_verified_claim_present_without_recall_yields_evidence_found(self) -> None:
        claims = [make_claim()]
        state = classify(claims, gold_recall=None, indicator_id="6.4", jurisdiction="SAMPLE")
        assert state == CoverageState.EVIDENCE_FOUND

    def test_no_claims_with_recall_yields_no_evidence_in_searched_corpus(self) -> None:
        state = classify([], gold_recall=0.82, indicator_id="6.4", jurisdiction="SAMPLE")
        assert state == CoverageState.NO_EVIDENCE_IN_SEARCHED_CORPUS

    def test_no_claims_without_recall_yields_insufficient_coverage(self) -> None:
        state = classify([], gold_recall=None, indicator_id="6.4", jurisdiction="SAMPLE")
        assert state == CoverageState.INSUFFICIENT_COVERAGE

    def test_only_non_verified_claims_with_recall_is_no_evidence(self) -> None:
        # PENDING / FLAGGED / REJECTED are filtered → behaves like an empty list.
        claims = [
            make_claim(claim_id="c1", layer1_status=Layer1Status.PENDING_VERIFICATION),
            make_claim(claim_id="c2", layer1_status=Layer1Status.FLAGGED),
            make_claim(claim_id="c3", layer1_status=Layer1Status.REJECTED),
        ]
        state = classify(claims, gold_recall=0.5, indicator_id="6.4", jurisdiction="SAMPLE")
        assert state == CoverageState.NO_EVIDENCE_IN_SEARCHED_CORPUS

    def test_only_non_verified_claims_without_recall_is_insufficient(self) -> None:
        claims = [
            make_claim(claim_id="c1", layer1_status=Layer1Status.PENDING_VERIFICATION),
            make_claim(claim_id="c2", layer1_status=Layer1Status.FLAGGED),
        ]
        state = classify(claims, gold_recall=None, indicator_id="6.4", jurisdiction="SAMPLE")
        assert state == CoverageState.INSUFFICIENT_COVERAGE

    def test_wrong_indicator_id_filtered_out(self) -> None:
        claims = [make_claim(indicator_id="7.1")]
        state = classify(claims, gold_recall=0.9, indicator_id="6.4", jurisdiction="SAMPLE")
        assert state == CoverageState.NO_EVIDENCE_IN_SEARCHED_CORPUS

    def test_wrong_jurisdiction_filtered_out(self) -> None:
        claims = [make_claim(jurisdiction="OTHER")]
        state = classify(claims, gold_recall=0.9, indicator_id="6.4", jurisdiction="SAMPLE")
        assert state == CoverageState.NO_EVIDENCE_IN_SEARCHED_CORPUS

    def test_bare_zero_is_structurally_impossible(self) -> None:
        # The truth table has exactly three outputs — none of them is a literal 0.
        for recall in (None, 0.0, 0.5, 1.0):
            for claims in ([], [make_claim()]):
                state = classify(
                    claims, gold_recall=recall, indicator_id="6.4", jurisdiction="SAMPLE"
                )
                assert state in {
                    CoverageState.EVIDENCE_FOUND,
                    CoverageState.NO_EVIDENCE_IN_SEARCHED_CORPUS,
                    CoverageState.INSUFFICIENT_COVERAGE,
                }


# ────────────────────────────────────────────────────────────────────────────
# filter_qualifying_claims()
# ────────────────────────────────────────────────────────────────────────────


class TestFilterQualifyingClaims:
    def test_only_verified_claims_kept(self) -> None:
        claims = [
            make_claim(claim_id="v1", layer1_status=Layer1Status.VERIFIED),
            make_claim(claim_id="p", layer1_status=Layer1Status.PENDING_VERIFICATION),
            make_claim(claim_id="f", layer1_status=Layer1Status.FLAGGED),
            make_claim(claim_id="r", layer1_status=Layer1Status.REJECTED),
            make_claim(claim_id="v2", layer1_status=Layer1Status.VERIFIED),
        ]
        kept = filter_qualifying_claims(claims, indicator_id="6.4", jurisdiction="SAMPLE")
        assert {c.claim_id for c in kept} == {"v1", "v2"}

    def test_indicator_mismatch_filtered(self) -> None:
        claims = [
            make_claim(claim_id="match", indicator_id="6.4"),
            make_claim(claim_id="other", indicator_id="7.1"),
        ]
        kept = filter_qualifying_claims(claims, indicator_id="6.4", jurisdiction="SAMPLE")
        assert [c.claim_id for c in kept] == ["match"]

    def test_jurisdiction_mismatch_filtered(self) -> None:
        claims = [
            make_claim(claim_id="ours", jurisdiction="SAMPLE"),
            make_claim(claim_id="theirs", jurisdiction="OTHER"),
        ]
        kept = filter_qualifying_claims(claims, indicator_id="6.4", jurisdiction="SAMPLE")
        assert [c.claim_id for c in kept] == ["ours"]

    def test_empty_input_returns_empty(self) -> None:
        assert filter_qualifying_claims([], indicator_id="6.4", jurisdiction="SAMPLE") == []


# ────────────────────────────────────────────────────────────────────────────
# miss_pct_from_recall() — rounding rule
# ────────────────────────────────────────────────────────────────────────────


class TestMissPctRounding:
    """Rule: ``int(round((1 - recall) * 100))`` — banker's rounding underneath.

    These cases pin the rule explicitly so the audit-package text is reproducible.
    """

    @pytest.mark.parametrize(
        ("recall", "expected_miss_pct"),
        [
            (1.0, 0),
            (0.99, 1),
            (0.95, 5),
            (0.90, 10),
            (0.82, 18),
            (0.5, 50),
            (0.18, 82),
            (0.01, 99),
            (0.0, 100),
        ],
    )
    def test_rounding_table(self, recall: float, expected_miss_pct: int) -> None:
        assert miss_pct_from_recall(recall) == expected_miss_pct

    @pytest.mark.parametrize("bad", [-0.01, 1.01, 2.0, -1.0])
    def test_out_of_range_rejected(self, bad: float) -> None:
        with pytest.raises(ValueError, match="must be in"):
            miss_pct_from_recall(bad)

    def test_implementation_matches_documented_formula(self) -> None:
        """The implementation IS ``int(round((1-recall)*100))`` — verify directly."""
        for recall in (0.0, 0.18, 0.5, 0.82, 0.95, 0.99, 1.0):
            assert miss_pct_from_recall(recall) == int(round((1.0 - recall) * 100.0))

    def test_bankers_rounding_on_exact_halves(self) -> None:
        """When the input is an exact ``*.5``, Python ``round()`` is banker's.

        We can't easily craft a recall that yields an exact half after float
        multiplication, but we can prove the underlying rule with exact halves
        on the percent scale: ``round(17.5) == 18`` and ``round(16.5) == 16``.
        """
        # Sanity-check the assumption the helper relies on.
        assert round(17.5) == 18
        assert round(16.5) == 16
        assert round(0.5) == 0


# ────────────────────────────────────────────────────────────────────────────
# format_no_evidence_reason()
# ────────────────────────────────────────────────────────────────────────────


class TestFormatNoEvidenceReason:
    def test_recall_082_mentions_18_percent(self) -> None:
        msg = format_no_evidence_reason(0.82)
        assert "18%" in msg
        assert "0.82" in msg
        assert "miss risk" in msg

    def test_recall_one_is_zero_miss(self) -> None:
        msg = format_no_evidence_reason(1.0)
        assert "0%" in msg

    def test_recall_zero_is_hundred_miss(self) -> None:
        msg = format_no_evidence_reason(0.0)
        assert "100%" in msg

    def test_rejects_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="must be in"):
            format_no_evidence_reason(1.5)


# ────────────────────────────────────────────────────────────────────────────
# format_insufficient_coverage_reason()
# ────────────────────────────────────────────────────────────────────────────


def test_insufficient_coverage_reason_is_stable() -> None:
    assert format_insufficient_coverage_reason() == (
        "no gold-set recall measured for this indicator"
    )


# ────────────────────────────────────────────────────────────────────────────
# requires_authoritative_source_check()
# ────────────────────────────────────────────────────────────────────────────


class TestRequiresAuthoritativeSourceCheck:
    def test_true_only_for_insufficient_coverage(self) -> None:
        assert requires_authoritative_source_check(CoverageState.INSUFFICIENT_COVERAGE) is True

    def test_false_for_evidence_found(self) -> None:
        assert requires_authoritative_source_check(CoverageState.EVIDENCE_FOUND) is False

    def test_false_for_no_evidence_in_searched_corpus(self) -> None:
        assert (
            requires_authoritative_source_check(CoverageState.NO_EVIDENCE_IN_SEARCHED_CORPUS)
            is False
        )
