"""End-to-end tests for deterministic Layer-2 score recommendations."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from rie_config import ConfigRepository
from rie_contracts import (
    Claim,
    ClausePattern,
    CoverageRecord,
    CoverageState,
    Decomposition,
    EvidenceSpan,
    IndicatorConfig,
    Layer1Status,
    Layer2Recommendation,
    LegalRegime,
    ScoreBand,
)
from rie_coverage import Layer2ScoringService
from rie_coverage.scoring_policy import recommend_band

from .conftest import make_claim

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def scorer() -> Layer2ScoringService:
    return Layer2ScoringService()


@pytest.fixture
def indicator_64() -> IndicatorConfig:
    config = ConfigRepository(repo_root=REPO_ROOT)
    pillar = config.load_pillar("6")
    ind = next(i for i in pillar.indicators if i.indicator_id == "6.4")
    return ind


@pytest.fixture
def evidence_coverage() -> CoverageRecord:
    return CoverageRecord(
        jurisdiction="SAMPLE",
        indicator_id="6.4",
        state=CoverageState.EVIDENCE_FOUND,
        verified_claim_ids=["claim_001"],
    )


class TestRecommendBand:
    def test_conditional_regime_recommends_half_band(
        self,
        make_claim_fixture,
        indicator_64: IndicatorConfig,
        evidence_coverage: CoverageRecord,
    ) -> None:
        claim = make_claim_fixture()
        band, rationale, questions = recommend_band(claim, indicator_64, evidence_coverage)

        assert band is ScoreBand.HALF
        assert "conditional_regime" in rationale
        assert "recommended band 0.5" in rationale
        assert any("exception" in q.lower() for q in questions)

    def test_human_confirmation_always_required(
        self,
        scorer: Layer2ScoringService,
        make_claim_fixture,
        indicator_64: IndicatorConfig,
        evidence_coverage: CoverageRecord,
    ) -> None:
        rec = scorer.recommend_for_claim(make_claim_fixture(), indicator_64, evidence_coverage)
        assert isinstance(rec, Layer2Recommendation)
        assert rec.human_confirmation_required is True

    def test_no_evidence_coverage_yields_null_band(
        self,
        make_claim_fixture,
        indicator_64: IndicatorConfig,
    ) -> None:
        coverage = CoverageRecord(
            jurisdiction="SAMPLE",
            indicator_id="6.4",
            state=CoverageState.NO_EVIDENCE_IN_SEARCHED_CORPUS,
            measured_recall=0.82,
            reason="no clauses passed verification; recall=0.82, miss risk ~18%",
        )
        band, rationale, _ = recommend_band(make_claim_fixture(), indicator_64, coverage)
        assert band is ScoreBand.NULL
        assert "no_evidence_in_searched_corpus" in rationale

    def test_insufficient_coverage_lists_corpus_question(
        self,
        make_claim_fixture,
        indicator_64: IndicatorConfig,
    ) -> None:
        coverage = CoverageRecord(
            jurisdiction="SAMPLE",
            indicator_id="6.4",
            state=CoverageState.INSUFFICIENT_COVERAGE,
            reason="no gold-set recall measured for this indicator",
        )
        _, _, questions = recommend_band(make_claim_fixture(), indicator_64, coverage)
        assert any("insufficient" in q.lower() or "widen" in q.lower() for q in questions)


class TestRecommendAll:
    def test_batch_produces_one_per_verified_claim(
        self,
        scorer: Layer2ScoringService,
        indicator_64: IndicatorConfig,
    ) -> None:
        config = ConfigRepository(repo_root=REPO_ROOT)
        pillar = config.load_pillar("6")
        claims = [make_claim(claim_id="c1")]
        coverage = CoverageRecord(
            jurisdiction="SAMPLE",
            indicator_id="6.4",
            state=CoverageState.EVIDENCE_FOUND,
            verified_claim_ids=["c1"],
        )
        recs = scorer.recommend_all(
            jurisdiction="SAMPLE",
            pillars=[pillar],
            verified_claims=claims,
            coverage_records=[coverage],
        )
        assert len(recs) == 1
        assert recs[0].claim_id == "c1"
        assert recs[0].recommended_band is ScoreBand.HALF

    def test_few_shot_example_influences_band(
        self,
        scorer: Layer2ScoringService,
    ) -> None:
        indicator = IndicatorConfig(
            indicator_id="99.1",
            name="Test indicator",
            definition="Test definition for scoring.",
            evaluation_method="clause_extraction",
            clause_pattern=ClausePattern.CONDITIONAL_REGIME,
            scoring_criteria={
                "0": "Fully prohibited.",
                "0.5": "Conditional regime.",
                "1": "Fully free.",
            },
            positive_keywords={"en": ["comparable level of protection", "shall not transfer"]},
            few_shot_examples=[
                {
                    "text": (
                        "An organisation shall not transfer any personal data outside the country "
                        "unless it has taken appropriate steps to ensure comparable protection."
                    ),
                    "decomposition": {
                        "subject": "organisation",
                        "condition": "transfer outside the country",
                        "constraint": "comparable-protection obligations",
                    },
                    "label": "99.1",
                    "score": 0.5,
                }
            ],
        )
        claim = Claim(
            claim_id="few_shot_claim",
            indicator_id="99.1",
            pillar_id="99",
            clause_id="elem_1",
            jurisdiction="SAMPLE",
            clause_pattern=ClausePattern.CONDITIONAL_REGIME,
            decomposition=Decomposition(
                subject="organisation",
                condition="transfer outside the country",
                constraint="comparable-protection obligations",
            ),
            evidence_spans=[
                EvidenceSpan(
                    span_id="doc1#0-10",
                    element_id="elem_1",
                    doc_id="doc1",
                    char_start=0,
                    char_end=10,
                )
            ],
            regime=LegalRegime(primary_element_id="elem_1", member_element_ids=["elem_1"]),
            layer1_status=Layer1Status.VERIFIED,
            created_at=datetime(2026, 5, 24, tzinfo=UTC),
        )
        coverage = CoverageRecord(
            jurisdiction="SAMPLE",
            indicator_id="99.1",
            state=CoverageState.EVIDENCE_FOUND,
            verified_claim_ids=["few_shot_claim"],
        )
        rec = scorer.recommend_for_claim(claim, indicator, coverage)
        assert rec.recommended_band is ScoreBand.HALF
