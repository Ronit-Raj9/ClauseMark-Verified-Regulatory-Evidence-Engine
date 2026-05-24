"""Unit tests for the RAGAS-style token-overlap proxies."""

from __future__ import annotations

import pytest
from rie_contracts import Decomposition
from rie_eval.ragas_metrics import (
    aggregate_recall,
    compute_context_recall,
    compute_faithfulness,
)

# ─── compute_context_recall ──────────────────────────────────────────────────


def test_context_recall_full_coverage() -> None:
    # Every gold token appears in the retrieved passage → 1.0.
    gold = "consent of the data subject"
    retrieved = ["An organisation shall obtain the consent of the data subject."]
    assert compute_context_recall(None, gold, retrieved) == pytest.approx(1.0)


def test_context_recall_zero_when_no_overlap() -> None:
    gold = "data protection commissioner"
    retrieved = ["completely unrelated text about widgets and gadgets"]
    assert compute_context_recall(None, gold, retrieved) == pytest.approx(0.0)


def test_context_recall_partial() -> None:
    # Gold has 4 distinct tokens: {consent, of, data, subject}.
    # Retrieved covers {consent, data} → 2/4 = 0.5.
    gold = "consent of data subject"
    retrieved = ["consent and data"]
    assert compute_context_recall(None, gold, retrieved) == pytest.approx(0.5)


def test_context_recall_empty_gold_is_zero() -> None:
    assert compute_context_recall(None, "", ["anything goes here"]) == pytest.approx(0.0)


def test_context_recall_empty_retrieved_is_zero() -> None:
    assert compute_context_recall(None, "consent", []) == pytest.approx(0.0)


def test_context_recall_normalises_case_and_punctuation() -> None:
    gold = "Consent, of THE data-subject!"
    retrieved = ["consent of the data subject"]
    # Underscored "data-subject" tokenises to {"data", "subject"} via \w+.
    assert compute_context_recall(None, gold, retrieved) == pytest.approx(1.0)


# ─── compute_faithfulness ────────────────────────────────────────────────────


def _decomp(constraint: str) -> Decomposition:
    return Decomposition(subject="organisation", constraint=constraint)


def test_faithfulness_identical_strings_full() -> None:
    decomp = _decomp("notify the commissioner within 72 hours")
    cited = "notify the commissioner within 72 hours"
    assert compute_faithfulness(decomp, cited) == pytest.approx(1.0)


def test_faithfulness_empty_constraint_is_zero() -> None:
    decomp = _decomp("")
    assert compute_faithfulness(decomp, "anything") == pytest.approx(0.0)


def test_faithfulness_empty_cited_is_zero() -> None:
    decomp = _decomp("notify the commissioner")
    assert compute_faithfulness(decomp, "") == pytest.approx(0.0)


def test_faithfulness_partial_overlap() -> None:
    # constraint distinct tokens: {notify, commissioner, within, 72, hours} = 5.
    # cited covers {notify, commissioner} = 2 → 0.4.
    decomp = _decomp("notify commissioner within 72 hours")
    cited = "the organisation shall notify the commissioner"
    assert compute_faithfulness(decomp, cited) == pytest.approx(0.4)


def test_faithfulness_disjoint_is_zero() -> None:
    decomp = _decomp("alpha beta gamma")
    cited = "delta epsilon zeta"
    assert compute_faithfulness(decomp, cited) == pytest.approx(0.0)


# ─── aggregate_recall ────────────────────────────────────────────────────────


def test_aggregate_recall_mean() -> None:
    assert aggregate_recall([1.0, 0.5, 0.0]) == pytest.approx(0.5)


def test_aggregate_recall_empty_zero() -> None:
    assert aggregate_recall([]) == pytest.approx(0.0)
