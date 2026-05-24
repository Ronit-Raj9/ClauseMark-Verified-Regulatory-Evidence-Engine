"""Unit tests for ``compute_hallucinated_words_rate``."""

from __future__ import annotations

import pytest
from rie_eval.hallucinated_words import compute_hallucinated_words_rate


def test_verbatim_match_is_zero() -> None:
    src = "The data subject shall provide consent."
    assert compute_hallucinated_words_rate(src, src) == pytest.approx(0.0)


def test_empty_extracted_is_zero() -> None:
    assert compute_hallucinated_words_rate("", "anything in source") == pytest.approx(0.0)


def test_single_foreign_word_yields_one_over_n() -> None:
    # "The data subject shall consent."  → 5 tokens.
    # Source missing "consent" → 1/5 = 0.2.
    extracted = "The data subject shall consent"
    source = "The data subject shall"
    assert compute_hallucinated_words_rate(extracted, source) == pytest.approx(1 / 5)


def test_all_foreign_yields_one() -> None:
    extracted = "alpha beta gamma"
    source = "completely different vocabulary"
    assert compute_hallucinated_words_rate(extracted, source) == pytest.approx(1.0)


def test_case_and_punctuation_normalised() -> None:
    extracted = "Consent, OF the data-subject!"
    source = "consent of the data subject is required."
    assert compute_hallucinated_words_rate(extracted, source) == pytest.approx(0.0)


def test_empty_source_yields_one_when_extracted_has_words() -> None:
    assert compute_hallucinated_words_rate("hello world", "") == pytest.approx(1.0)
