"""Tests for authoritative-source reachability + the defensible-0 truth table."""

from __future__ import annotations

import pytest
from rie_contracts import CoverageState
from rie_coverage.authoritative_sources import (
    AuthoritativeSourceCheck,
    assess_reachability,
    is_defensible_zero,
)

# ── assess_reachability ──────────────────────────────────────────────────────


def test_reachability_full() -> None:
    assert assess_reachability(["primary_law", "regulation"], ["primary_law", "regulation"]) == 1.0


def test_reachability_partial() -> None:
    assert assess_reachability(["a", "b", "c", "d"], ["a", "b"]) == 0.5


def test_reachability_empty_expected_is_one() -> None:
    assert assess_reachability([], ["anything"]) == 1.0


def test_reachability_set_based_ignores_duplicates_and_extras() -> None:
    # Duplicate + unexpected reached entries do not inflate the score.
    assert assess_reachability(["a", "b"], ["a", "a", "z"]) == 0.5


def test_reachability_none_reached() -> None:
    assert assess_reachability(["a", "b"], []) == 0.0


# ── is_defensible_zero truth table ───────────────────────────────────────────

NO_EV = CoverageState.NO_EVIDENCE_IN_SEARCHED_CORPUS


def test_defensible_when_high_recall_and_high_reach() -> None:
    assert is_defensible_zero(NO_EV, reachability=0.95, gold_recall=0.9) is True


def test_not_defensible_when_low_recall() -> None:
    assert is_defensible_zero(NO_EV, reachability=0.95, gold_recall=0.5) is False


def test_not_defensible_when_low_reach() -> None:
    assert is_defensible_zero(NO_EV, reachability=0.5, gold_recall=0.9) is False


def test_not_defensible_when_both_low() -> None:
    assert is_defensible_zero(NO_EV, reachability=0.2, gold_recall=0.2) is False


def test_not_defensible_when_recall_none() -> None:
    assert is_defensible_zero(NO_EV, reachability=1.0, gold_recall=None) is False


def test_floors_are_inclusive() -> None:
    assert is_defensible_zero(NO_EV, reachability=0.9, gold_recall=0.8) is True


@pytest.mark.parametrize(
    "state",
    [CoverageState.EVIDENCE_FOUND, CoverageState.INSUFFICIENT_COVERAGE],
)
def test_only_no_evidence_can_be_defensible(state: CoverageState) -> None:
    # No other state can ever be a defensible 0, regardless of the numbers.
    assert is_defensible_zero(state, reachability=1.0, gold_recall=1.0) is False


def test_custom_floors() -> None:
    assert is_defensible_zero(NO_EV, 0.7, 0.6, recall_floor=0.6, reach_floor=0.7) is True
    assert is_defensible_zero(NO_EV, 0.7, 0.6, recall_floor=0.6, reach_floor=0.71) is False


# ── AuthoritativeSourceCheck dataclass ───────────────────────────────────────


def test_check_derives_reachability_from_counts() -> None:
    chk = AuthoritativeSourceCheck(
        indicator_id="6.4",
        jurisdiction="SAMPLE",
        expected_source_types=["primary_law", "regulation"],
        sources_reached=1,
        sources_total=2,
    )
    assert chk.reachability == 0.5


def test_check_explicit_reachability_preserved() -> None:
    chk = AuthoritativeSourceCheck(
        indicator_id="6.4",
        jurisdiction="SAMPLE",
        sources_reached=1,
        sources_total=2,
        reachability=0.42,
    )
    assert chk.reachability == 0.42
