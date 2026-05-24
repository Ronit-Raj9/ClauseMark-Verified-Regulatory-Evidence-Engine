"""Retrieval-recall tests for `Evaluator.measure_retrieval_recall`."""

from __future__ import annotations

import pytest
from rie_config import ConfigRepository
from rie_eval import Evaluator


def test_recall_perfect_when_doc_id_in_every_retrieved_list(
    config: ConfigRepository,
) -> None:
    # Pillar 7 has 3 gold items, all touching sample_dpa_2020.
    retrieved = [
        ["sample_dpa_2020_s6", "x", "y"],
        ["sample_dpa_2020_s33", "x"],
        ["sample_dpa_2020_s7"],
    ]
    evaluator = Evaluator(config=config)
    assert evaluator.measure_retrieval_recall("7", retrieved) == pytest.approx(1.0)


def test_recall_zero_when_nothing_matches(config: ConfigRepository) -> None:
    retrieved = [["foo"], ["bar"], ["baz"]]
    evaluator = Evaluator(config=config)
    assert evaluator.measure_retrieval_recall("7", retrieved) == pytest.approx(0.0)


def test_recall_partial(config: ConfigRepository) -> None:
    # 2 of 3 retrieve the right doc.
    retrieved = [
        ["sample_dpa_2020_s6"],
        ["mismatch"],
        ["sample_dpa_2020_s7"],
    ]
    evaluator = Evaluator(config=config)
    assert evaluator.measure_retrieval_recall("7", retrieved) == pytest.approx(2 / 3)


def test_recall_via_span_text_substring(config: ConfigRepository) -> None:
    # Caller supplies plain-text snippets. The gold span_text appears as a
    # normalised substring of the snippet — count as recalled.
    retrieved = [
        [
            "An organisation shall only process personal data with the "
            "consent of the data subject or on another lawful basis "
            "specified in section 6."
        ],
        [
            "A controller shall notify the Commissioner without undue "
            "delay, and in any event within 72 hours, of any personal "
            "data breach, and shall notify affected data subjects where "
            "the breach is likely to result in a high risk to their rights."
        ],
        [
            "The Data Protection Commission shall be independent in the "
            "exercise of its functions and shall not be subject to the "
            "direction or control of any person or authority."
        ],
    ]
    evaluator = Evaluator(config=config)
    assert evaluator.measure_retrieval_recall("7", retrieved) == pytest.approx(1.0)


def test_recall_mismatched_length_raises(config: ConfigRepository) -> None:
    evaluator = Evaluator(config=config)
    with pytest.raises(ValueError, match="retrieved_per_query length"):
        evaluator.measure_retrieval_recall("7", [["foo"]])


def test_recall_empty_gold_returns_zero(config: ConfigRepository) -> None:
    # Pillar 8 has no gold items shipped.
    evaluator = Evaluator(config=config)
    assert evaluator.measure_retrieval_recall("8", []) == 0.0
