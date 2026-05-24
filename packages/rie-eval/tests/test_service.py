"""Tests for `Evaluator.evaluate_pillar` against the real gold sets + a
synthetic claim stream + a tiny in-memory fake repository."""

from __future__ import annotations

import pytest
from rie_config import ConfigRepository
from rie_contracts import (
    AuthorityTier,
    ClausePattern,
    CoverageState,
    EvaluatorPort,
    Layer1Status,
    ReviewDecision,
)
from rie_eval import Evaluator

from .conftest import (
    FakeRepo,
    make_claim,
    make_coverage,
    make_doc_meta,
    make_review,
)


def test_evaluator_implements_port(config: ConfigRepository) -> None:
    """`@runtime_checkable` Protocol → an `isinstance` check is the contract."""
    evaluator = Evaluator(config=config)
    assert isinstance(evaluator, EvaluatorPort)


def test_perfect_predictions_pillar_07(config: ConfigRepository) -> None:
    # Pillar 7 has three gold items: 7.2 (obligation), 7.4, 7.5.
    claims = [
        make_claim(
            claim_id="c1",
            indicator_id="7.2",
            pillar_id="7",
            clause_pattern=ClausePattern.OBLIGATION,
            char_start=100,
        ),
        make_claim(
            claim_id="c2",
            indicator_id="7.4",
            pillar_id="7",
            clause_pattern=ClausePattern.OBLIGATION,
            char_start=200,
        ),
        make_claim(
            claim_id="c3",
            indicator_id="7.5",
            pillar_id="7",
            clause_pattern=ClausePattern.OBLIGATION,
            char_start=300,
        ),
    ]
    evaluator = Evaluator(config=config)
    out = evaluator.evaluate_pillar("7", claims)
    assert out["precision_macro"] == pytest.approx(1.0)
    assert out["recall_macro"] == pytest.approx(1.0)
    assert out["f1_macro"] == pytest.approx(1.0)
    assert out["count_total"] == 3.0
    assert out["count_verified"] == 3.0
    # No reviews supplied, in-memory fallback → 1.0 per spec.
    assert out["citation_support_precision"] == pytest.approx(1.0)


def test_recall_drops_when_a_gold_item_is_missed(config: ConfigRepository) -> None:
    # Drop the 7.5 claim: gold expects three, system emits two.
    claims = [
        make_claim(claim_id="c1", indicator_id="7.2", pillar_id="7", char_start=100),
        make_claim(claim_id="c2", indicator_id="7.4", pillar_id="7", char_start=200),
    ]
    evaluator = Evaluator(config=config)
    out = evaluator.evaluate_pillar("7", claims)
    # 7.5 had 1 gold + 0 predictions → recall[7.5] == 0
    assert out["recall[7.5]"] == pytest.approx(0.0)
    # Macro recall now strictly less than 1.0
    assert out["recall_macro"] < 1.0


def test_indicator_swap_hits_precision_and_recall(
    config: ConfigRepository,
) -> None:
    # Swap 7.2 prediction → 7.4 (collides with existing 7.4 gold). System
    # also still emits a correct 7.4. Net: 7.2 missed, two 7.4 predictions
    # but only one true 7.4 gold so one is a false positive.
    claims = [
        make_claim(claim_id="c1", indicator_id="7.4", pillar_id="7", char_start=100),
        make_claim(claim_id="c2", indicator_id="7.4", pillar_id="7", char_start=200),
        make_claim(claim_id="c3", indicator_id="7.5", pillar_id="7", char_start=300),
    ]
    evaluator = Evaluator(config=config)
    out = evaluator.evaluate_pillar("7", claims)
    # 7.2: no prediction → P/R/F1 = 0
    assert out["recall[7.2]"] == pytest.approx(0.0)
    # 7.4: tp=1, fp=1 → precision = 0.5
    assert out["precision[7.4]"] == pytest.approx(0.5)
    assert out["recall[7.4]"] == pytest.approx(1.0)


def test_citation_support_precision_from_reviews(
    config: ConfigRepository, fake_repo: FakeRepo
) -> None:
    # Two verified claims, one accepted by reviewer, one rejected.
    claims = [
        make_claim(
            claim_id="c1",
            indicator_id="7.2",
            pillar_id="7",
            status=Layer1Status.VERIFIED,
            char_start=100,
        ),
        make_claim(
            claim_id="c2",
            indicator_id="7.4",
            pillar_id="7",
            status=Layer1Status.VERIFIED,
            char_start=200,
        ),
    ]
    fake_repo.reviews = {
        "c1": [make_review(claim_id="c1", decision=ReviewDecision.ACCEPT)],
        "c2": [make_review(claim_id="c2", decision=ReviewDecision.REJECT)],
    }
    fake_repo.documents = {"sample_dpa_2020": make_doc_meta()}
    evaluator = Evaluator(config=config, repo=fake_repo)
    out = evaluator.evaluate_pillar("7", claims)
    assert out["citation_support_precision"] == pytest.approx(0.5)


def test_citation_support_precision_falls_back_to_one(
    config: ConfigRepository,
) -> None:
    # No repo provided → fallback per spec.
    claims = [
        make_claim(
            claim_id="c1",
            indicator_id="7.2",
            pillar_id="7",
            status=Layer1Status.VERIFIED,
            char_start=100,
        ),
    ]
    evaluator = Evaluator(config=config)
    out = evaluator.evaluate_pillar("7", claims)
    assert out["citation_support_precision"] == pytest.approx(1.0)


def test_authority_tier_error_rate_detects_mismatch(
    config: ConfigRepository, fake_repo: FakeRepo
) -> None:
    # Gold expects TIER_1_STATUTE; stored doc has TIER_3_GUIDELINE → 100% error.
    fake_repo.documents = {
        "sample_dpa_2020": make_doc_meta(authority_tier=AuthorityTier.TIER_3_GUIDELINE)
    }
    claims = [
        make_claim(claim_id="c1", indicator_id="7.2", pillar_id="7", char_start=100),
        make_claim(claim_id="c2", indicator_id="7.4", pillar_id="7", char_start=200),
        make_claim(claim_id="c3", indicator_id="7.5", pillar_id="7", char_start=300),
    ]
    evaluator = Evaluator(config=config, repo=fake_repo)
    out = evaluator.evaluate_pillar("7", claims)
    assert out["authority_tier_error_rate"] == pytest.approx(1.0)


def test_authority_tier_error_rate_zero_on_matching_tier(
    config: ConfigRepository, fake_repo: FakeRepo
) -> None:
    fake_repo.documents = {
        "sample_dpa_2020": make_doc_meta(authority_tier=AuthorityTier.TIER_1_STATUTE)
    }
    claims = [
        make_claim(claim_id="c1", indicator_id="7.2", pillar_id="7", char_start=100),
    ]
    evaluator = Evaluator(config=config, repo=fake_repo)
    out = evaluator.evaluate_pillar("7", claims)
    assert out["authority_tier_error_rate"] == pytest.approx(0.0)


def test_false_zero_rate_detects_silent_zero(config: ConfigRepository, fake_repo: FakeRepo) -> None:
    # System emitted a no_evidence row for 7.2 in SAMPLE, but gold has a 7.2.
    fake_repo.coverage = [
        make_coverage(
            jurisdiction="SAMPLE",
            indicator_id="7.2",
            state=CoverageState.NO_EVIDENCE_IN_SEARCHED_CORPUS,
        ),
        # And an honest no_evidence for an indicator gold doesn't have.
        make_coverage(
            jurisdiction="SAMPLE",
            indicator_id="7.99",
            state=CoverageState.NO_EVIDENCE_IN_SEARCHED_CORPUS,
        ),
    ]
    fake_repo.documents = {"sample_dpa_2020": make_doc_meta()}
    evaluator = Evaluator(config=config, repo=fake_repo)
    out = evaluator.evaluate_pillar("7", [])
    # 1 false-zero out of 2 no_evidence rows = 0.5
    assert out["false_zero_rate"] == pytest.approx(0.5)


def test_false_zero_rate_zero_with_no_coverage_rows(
    config: ConfigRepository, fake_repo: FakeRepo
) -> None:
    evaluator = Evaluator(config=config, repo=fake_repo)
    out = evaluator.evaluate_pillar("7", [])
    assert out["false_zero_rate"] == pytest.approx(0.0)


def test_counts_track_layer1_status(config: ConfigRepository) -> None:
    claims = [
        make_claim(
            claim_id="c1",
            indicator_id="7.2",
            pillar_id="7",
            status=Layer1Status.VERIFIED,
        ),
        make_claim(
            claim_id="c2",
            indicator_id="7.4",
            pillar_id="7",
            status=Layer1Status.FLAGGED,
        ),
        make_claim(
            claim_id="c3",
            indicator_id="7.5",
            pillar_id="7",
            status=Layer1Status.REJECTED,
        ),
        make_claim(
            claim_id="c4",
            indicator_id="7.5",
            pillar_id="7",
            status=Layer1Status.VERIFIED,
        ),
    ]
    evaluator = Evaluator(config=config)
    out = evaluator.evaluate_pillar("7", claims)
    assert out["count_total"] == 4.0
    assert out["count_verified"] == 2.0
    assert out["count_flagged"] == 1.0
    assert out["count_rejected"] == 1.0


def test_other_pillar_claims_are_ignored(config: ConfigRepository) -> None:
    # Pillar 7 evaluator should ignore pillar 6 claims even if passed in.
    claims = [
        make_claim(claim_id="c1", indicator_id="6.4", pillar_id="6"),
        make_claim(claim_id="c2", indicator_id="7.2", pillar_id="7", char_start=100),
    ]
    evaluator = Evaluator(config=config)
    out = evaluator.evaluate_pillar("7", claims)
    assert out["count_total"] == 1.0


def test_reviewer_override_rate_keyed_into_output(config: ConfigRepository) -> None:
    # Without a repo, reviewer override rate is reported as 0.0.
    claims = [
        make_claim(claim_id="c1", indicator_id="7.2", pillar_id="7", char_start=100),
    ]
    evaluator = Evaluator(config=config)
    out = evaluator.evaluate_pillar("7", claims)
    assert "reviewer_override_rate" in out
    assert out["reviewer_override_rate"] == pytest.approx(0.0)


def test_reviewer_override_rate_counts_correct_and_reject(
    config: ConfigRepository, fake_repo: FakeRepo
) -> None:
    # 3 verified claims: 1 ACCEPT, 1 CORRECT, 1 REJECT → 2/3 override rate.
    claims = [
        make_claim(
            claim_id="c1",
            indicator_id="7.2",
            pillar_id="7",
            status=Layer1Status.VERIFIED,
            char_start=100,
        ),
        make_claim(
            claim_id="c2",
            indicator_id="7.4",
            pillar_id="7",
            status=Layer1Status.VERIFIED,
            char_start=200,
        ),
        make_claim(
            claim_id="c3",
            indicator_id="7.5",
            pillar_id="7",
            status=Layer1Status.VERIFIED,
            char_start=300,
        ),
    ]
    fake_repo.reviews = {
        "c1": [make_review(claim_id="c1", decision=ReviewDecision.ACCEPT)],
        "c2": [make_review(claim_id="c2", decision=ReviewDecision.CORRECT)],
        "c3": [make_review(claim_id="c3", decision=ReviewDecision.REJECT)],
    }
    fake_repo.documents = {"sample_dpa_2020": make_doc_meta()}
    evaluator = Evaluator(config=config, repo=fake_repo)
    out = evaluator.evaluate_pillar("7", claims)
    assert out["reviewer_override_rate"] == pytest.approx(2 / 3)


def test_context_recall_and_faithfulness_keys_present(config: ConfigRepository) -> None:
    """RAGAS-proxy metrics surface in evaluate_pillar output."""
    claims = [
        make_claim(claim_id="c1", indicator_id="7.2", pillar_id="7", char_start=100),
        make_claim(claim_id="c2", indicator_id="7.4", pillar_id="7", char_start=200),
        make_claim(claim_id="c3", indicator_id="7.5", pillar_id="7", char_start=300),
    ]
    evaluator = Evaluator(config=config)
    out = evaluator.evaluate_pillar("7", claims)
    assert "context_recall" in out
    assert "faithfulness" in out
    # Without a repo, cited text falls back to the gold span text →
    # context_recall is 1.0 by construction.
    assert out["context_recall"] == pytest.approx(1.0)
    # Faithfulness lives in [0, 1] — exact value depends on the synthetic
    # decomposition's token overlap with the gold span text.
    assert 0.0 <= out["faithfulness"] <= 1.0


def test_detailed_per_indicator_returns_grouped_view(
    config: ConfigRepository,
) -> None:
    claims = [
        make_claim(claim_id="c1", indicator_id="7.2", pillar_id="7", char_start=100),
        make_claim(claim_id="c2", indicator_id="7.4", pillar_id="7", char_start=200),
        make_claim(claim_id="c3", indicator_id="7.5", pillar_id="7", char_start=300),
    ]
    evaluator = Evaluator(config=config)
    grouped = evaluator.detailed_per_indicator("7", claims)
    assert set(grouped) == {
        "precision_by_indicator",
        "recall_by_indicator",
        "f1_by_indicator",
    }
    assert grouped["f1_by_indicator"]["7.2"] == pytest.approx(1.0)
    assert grouped["f1_by_indicator"]["7.4"] == pytest.approx(1.0)
    assert grouped["f1_by_indicator"]["7.5"] == pytest.approx(1.0)
