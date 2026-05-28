"""Tests for gold-set retrieval recall (§7 honest absence)."""

from __future__ import annotations

import pytest
from rie_config import ConfigRepository
from rie_contracts import (
    AuthorityTier,
    ClausePattern,
    GoldItem,
    ScoreBand,
)
from rie_eval import Evaluator, compute_indicator_gold_recall, gold_hit_in_retrieved


def _gold_item(
    *,
    gold_id: str = "g1",
    indicator_id: str = "6.4",
    jurisdiction: str = "SAMPLE",
    doc_id: str = "sample_dpa_2020",
    span_text: str = "transfer personal data outside",
) -> GoldItem:
    return GoldItem(
        gold_id=gold_id,
        doc_id=doc_id,
        jurisdiction=jurisdiction,
        pillar_id="6",
        indicator_id=indicator_id,
        span_text=span_text,
        expected_clause_pattern=ClausePattern.CONDITIONAL_REGIME,
        expected_score_band=ScoreBand.HALF,
        expected_authority_tier=AuthorityTier.TIER_1_STATUTE,
    )


def test_gold_hit_by_doc_id_substring() -> None:
    item = _gold_item()
    assert gold_hit_in_retrieved(item, ["sample_dpa_2020_s26"]) is True


def test_gold_hit_by_span_text_substring() -> None:
    item = _gold_item(span_text="comparable protection under this Act")
    assert gold_hit_in_retrieved(item, ["comparable protection under this Act"]) is True


def test_gold_hit_false_when_no_overlap() -> None:
    item = _gold_item()
    assert gold_hit_in_retrieved(item, ["other_doc_s1"]) is False


def test_compute_indicator_gold_recall_perfect() -> None:
    gold = [_gold_item(gold_id="g1"), _gold_item(gold_id="g2")]
    recall = compute_indicator_gold_recall(
        gold,
        indicator_id="6.4",
        jurisdiction="SAMPLE",
        retrieved=["sample_dpa_2020_s26", "sample_dpa_2020_s27"],
    )
    assert recall == pytest.approx(1.0)


def test_compute_indicator_gold_recall_partial() -> None:
    gold = [
        _gold_item(gold_id="g1", doc_id="sample_dpa_2020"),
        _gold_item(
            gold_id="g2",
            doc_id="other_statute",
            span_text="servers located within the territory",
        ),
    ]
    recall = compute_indicator_gold_recall(
        gold,
        indicator_id="6.4",
        jurisdiction="SAMPLE",
        retrieved=["sample_dpa_2020_s26"],
    )
    assert recall == pytest.approx(0.5)


def test_compute_indicator_gold_recall_none_without_gold() -> None:
    gold = [_gold_item(indicator_id="6.3")]
    assert (
        compute_indicator_gold_recall(
            gold,
            indicator_id="6.4",
            jurisdiction="SAMPLE",
            retrieved=["sample_dpa_2020_s26"],
        )
        is None
    )


def test_compute_indicator_gold_recall_none_without_retrieval() -> None:
    gold = [_gold_item()]
    assert (
        compute_indicator_gold_recall(
            gold,
            indicator_id="6.4",
            jurisdiction="SAMPLE",
            retrieved=[],
        )
        is None
    )


def test_evaluator_lookup_gold_recall_uses_real_gold(config: ConfigRepository) -> None:
    evaluator = Evaluator(config=config)
    recall = evaluator.lookup_gold_recall(
        "6",
        "6.4",
        "SAMPLE",
        retrieved=["sample_dpa_2020_s26"],
    )
    assert recall is not None
    assert recall == pytest.approx(1.0)


def test_evaluator_lookup_gold_recall_none_for_unknown_indicator(
    config: ConfigRepository,
) -> None:
    evaluator = Evaluator(config=config)
    assert (
        evaluator.lookup_gold_recall(
            "6",
            "6.99",
            "SAMPLE",
            retrieved=["sample_dpa_2020_s26"],
        )
        is None
    )


def test_evaluator_lookup_gold_recall_matches_orchestration_run_hits(
    config: ConfigRepository,
) -> None:
    """Same inputs orchestration passes from flattened retrieval hits."""
    evaluator = Evaluator(config=config)
    retrieved = [
        "sample_dpa_2020_s26",
        "sample_dpa_2020_s26_chunk0",
    ]
    recall = evaluator.lookup_gold_recall("6", "6.4", "SAMPLE", retrieved)
    assert recall is not None
    assert 0.0 <= recall <= 1.0
