"""Tests for size-weighted cross-corpus recall aggregation."""

from __future__ import annotations

import pytest
from rie_coverage.cross_corpus import CorpusRecall, CrossCorpusRecall


def test_weighted_combine_favours_larger_corpus() -> None:
    recalls = [
        CorpusRecall(corpus_id="big", recall=0.5, size=900),
        CorpusRecall(corpus_id="small", recall=1.0, size=100),
    ]
    # (0.5*900 + 1.0*100) / 1000 = 0.55 — the small well-covered corpus cannot
    # mask the large poorly-covered one.
    assert CrossCorpusRecall.combine(recalls) == pytest.approx(0.55)


def test_equal_sizes_is_plain_mean() -> None:
    recalls = [
        CorpusRecall(corpus_id="a", recall=0.8, size=50),
        CorpusRecall(corpus_id="b", recall=0.6, size=50),
    ]
    assert CrossCorpusRecall.combine(recalls) == pytest.approx(0.7)


def test_zero_size_corpus_contributes_nothing() -> None:
    recalls = [
        CorpusRecall(corpus_id="real", recall=0.9, size=100),
        CorpusRecall(corpus_id="empty", recall=0.0, size=0),
    ]
    assert CrossCorpusRecall.combine(recalls) == pytest.approx(0.9)


def test_empty_list_is_zero() -> None:
    assert CrossCorpusRecall.combine([]) == 0.0


def test_all_zero_size_is_zero() -> None:
    recalls = [CorpusRecall(corpus_id="x", recall=0.9, size=0)]
    assert CrossCorpusRecall.combine(recalls) == 0.0


def test_combined_recall_method() -> None:
    agg = CrossCorpusRecall(
        indicator_id="6.4",
        per_corpus=[
            CorpusRecall(corpus_id="a", recall=0.5, size=900),
            CorpusRecall(corpus_id="b", recall=1.0, size=100),
        ],
    )
    assert agg.combined_recall() == pytest.approx(0.55)


def test_result_stays_in_bounds() -> None:
    recalls = [CorpusRecall(corpus_id="a", recall=1.0, size=10)]
    assert 0.0 <= CrossCorpusRecall.combine(recalls) <= 1.0


def test_invalid_recall_rejected() -> None:
    with pytest.raises(ValueError):
        CorpusRecall(corpus_id="bad", recall=1.5, size=10)


def test_invalid_size_rejected() -> None:
    with pytest.raises(ValueError):
        CorpusRecall(corpus_id="bad", recall=0.5, size=-1)
