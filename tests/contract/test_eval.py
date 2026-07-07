"""Contract test for the EvaluatorPort.

Verifies any conforming implementation satisfies the runtime-checkable
``EvaluatorPort`` protocol and that the canonical metric keys produced by
``evaluate_pillar`` are present.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
from rie_config import ConfigRepository
from rie_contracts import Claim, EvaluatorPort
from rie_eval import Evaluator

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def config() -> ConfigRepository:
    return ConfigRepository(repo_root=REPO_ROOT)


@pytest.fixture
def evaluator(config: ConfigRepository) -> Evaluator:
    return Evaluator(config=config)


def test_evaluator_satisfies_port(evaluator: Evaluator) -> None:
    assert isinstance(evaluator, EvaluatorPort)


def test_evaluate_pillar_returns_canonical_metric_keys(evaluator: Evaluator) -> None:
    """The §8 metric surface must include every advertised key, even when
    the caller supplies an empty claim list."""
    claims: Sequence[Claim] = []
    out = evaluator.evaluate_pillar("7", claims)
    for key in (
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "citation_support_precision",
        "authority_tier_error_rate",
        "false_zero_rate",
        "reviewer_override_rate",
        "context_recall",
        "faithfulness",
        "count_total",
        "count_verified",
        "count_flagged",
        "count_rejected",
    ):
        assert key in out, f"missing metric key: {key}"
        assert isinstance(out[key], float), f"{key} not a float"


def test_measure_retrieval_recall_returns_float_in_range(
    evaluator: Evaluator, config: ConfigRepository
) -> None:
    # One retrieved list per gold item (live gold size varies as the real
    # Round-1 corpus grows, so size the input to the actual gold count).
    n_gold = len(list(config.load_gold("7")))
    retrieved = [["nothing"]] * n_gold
    score = evaluator.measure_retrieval_recall("7", retrieved)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
