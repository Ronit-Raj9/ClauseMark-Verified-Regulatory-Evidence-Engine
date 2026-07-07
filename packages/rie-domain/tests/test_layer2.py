"""Tests for Layer-2 recommendation helpers in ``rie_domain.layer2``."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from rie_contracts import (
    Claim,
    ClausePattern,
    CoverageRecord,
    CoverageState,
    Decomposition,
    Element,
    ElementType,
    EvaluationMethod,
    EvidenceSpan,
    IndicatorConfig,
    Layer1Status,
    LegalRegime,
    PillarConfig,
    ScoreBand,
    StructureEdge,
    StructureEdgeType,
)
from rie_contracts.models import DocumentProfile
from rie_domain import (
    assemble_regime_from_claim,
    build_layer2_batch,
    build_layer2_from_pillar,
    build_layer2_recommendation,
    claim_text,
    derive_layer1_status,
    enrich_claim_regime,
    find_coverage_for_claim,
    pick_band,
    recommend_band,
    regime_open_questions,
    score_from_few_shots,
)


def _indicator_64() -> IndicatorConfig:
    return IndicatorConfig(
        indicator_id="6.4",
        name="Conditional cross-border transfer regime (adequacy / safeguards)",
        definition="Conditional outbound transfer regime.",
        evaluation_method=EvaluationMethod.CLAUSE_EXTRACTION,
        clause_pattern=ClausePattern.CONDITIONAL_REGIME,
        scoring_criteria={
            "0": "No conditional regime — either fully free or fully prohibited.",
            "0.5": "Conditions exist but are narrow or sectoral.",
            "1": "A general conditional regime is the operating rule for outbound transfers.",
        },
        few_shot_examples=[
            {
                "text": (
                    "An organisation shall not transfer any personal data outside the country "
                    "unless it has taken appropriate steps to ensure comparable protection."
                ),
                "decomposition": {
                    "subject": "organisation",
                    "condition": "transfer outside the country",
                    "constraint": "recipient bound by comparable-protection obligations",
                },
                "label": "6.4",
                "score": 0.5,
            },
        ],
    )


def _pillar_6() -> PillarConfig:
    return PillarConfig(
        pillar_id="6",
        pillar_name="Cross-border data",
        cluster="digital_governance",
        document_profile=DocumentProfile.STATUTORY_LEGAL_TEXT,
        status="built",
        indicators=[_indicator_64()],
    )


def _claim(*, claim_id: str = "claim_001", regime: LegalRegime | None = None) -> Claim:
    return Claim(
        claim_id=claim_id,
        indicator_id="6.4",
        pillar_id="6",
        clause_id="elem_1",
        jurisdiction="SAMPLE",
        clause_pattern=ClausePattern.CONDITIONAL_REGIME,
        decomposition=Decomposition(
            subject="organisation",
            condition="transfer outside the country",
            constraint="recipient bound by comparable-protection obligations",
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
        regime=regime or LegalRegime(primary_element_id="elem_1", member_element_ids=["elem_1"]),
        layer1_status=Layer1Status.VERIFIED,
        created_at=datetime(2026, 5, 24, tzinfo=UTC),
    )


@pytest.fixture
def indicator_64() -> IndicatorConfig:
    return _indicator_64()


@pytest.fixture
def evidence_coverage() -> CoverageRecord:
    return CoverageRecord(
        jurisdiction="SAMPLE",
        indicator_id="6.4",
        state=CoverageState.EVIDENCE_FOUND,
        verified_claim_ids=["claim_001"],
    )


def test_build_layer2_recommendation_requires_human_confirmation(
    indicator_64: IndicatorConfig,
    evidence_coverage: CoverageRecord,
) -> None:
    rec = build_layer2_recommendation(_claim(), indicator_64, evidence_coverage)
    assert rec.human_confirmation_required is True
    assert rec.recommended_band is ScoreBand.HALF
    assert rec.claim_id == "claim_001"
    assert rec.indicator_id == "6.4"


def test_recommend_band_null_when_no_evidence(
    indicator_64: IndicatorConfig,
) -> None:
    coverage = CoverageRecord(
        jurisdiction="SAMPLE",
        indicator_id="6.4",
        state=CoverageState.NO_EVIDENCE_IN_SEARCHED_CORPUS,
        measured_recall=0.82,
    )
    band, rationale, _ = recommend_band(_claim(), indicator_64, coverage)
    assert band is ScoreBand.NULL
    assert "no_evidence_in_searched_corpus" in rationale


def test_build_layer2_from_pillar_resolves_indicator(
    evidence_coverage: CoverageRecord,
) -> None:
    rec = build_layer2_from_pillar(_claim(), _pillar_6(), evidence_coverage)
    assert rec is not None
    assert rec.recommended_band is ScoreBand.HALF


def test_build_layer2_from_pillar_returns_none_for_unknown_indicator(
    evidence_coverage: CoverageRecord,
) -> None:
    wrong = _claim().model_copy(update={"indicator_id": "99.9"})
    assert build_layer2_from_pillar(wrong, _pillar_6(), evidence_coverage) is None


def test_assemble_regime_from_claim_uses_primary_element() -> None:
    elements = {
        eid: Element(
            element_id=eid,
            doc_id="d1",
            element_type=ElementType.PARAGRAPH,
            text="x",
            page=1,
            char_start=0,
            char_end=1,
            extraction_confidence=1.0,
        )
        for eid in ("primary", "defn")
    }
    edges = [
        StructureEdge(
            from_element="primary", to_element="defn", edge_type=StructureEdgeType.DEFINES
        ),
    ]
    claim = _claim(regime=LegalRegime(primary_element_id="primary", member_element_ids=["primary"]))
    regime = assemble_regime_from_claim(claim, elements, edges, max_depth=1)
    assert "defn" in regime.definitions
    assert regime.primary_element_id == "primary"


def test_assemble_regime_from_claim_falls_back_to_clause_id() -> None:
    elements = {
        "elem_1": Element(
            element_id="elem_1",
            doc_id="d1",
            element_type=ElementType.PARAGRAPH,
            text="x",
            page=1,
            char_start=0,
            char_end=1,
            extraction_confidence=1.0,
        ),
    }
    claim = _claim(
        regime=LegalRegime(primary_element_id="missing", member_element_ids=["missing"]),
    )
    regime = assemble_regime_from_claim(claim, elements, [], max_depth=0)
    assert regime.primary_element_id == "elem_1"


def test_derive_layer1_status_paths() -> None:
    assert derive_layer1_status(False, False) == Layer1Status.REJECTED
    assert derive_layer1_status(True, True) == Layer1Status.FLAGGED
    assert derive_layer1_status(True, False) == Layer1Status.PENDING_VERIFICATION


def test_claim_text_flattens_decomposition() -> None:
    claim = _claim()
    text = claim_text(claim)
    assert "organisation" in text
    assert "comparable-protection" in text
    assert ClausePattern.CONDITIONAL_REGIME.value in text


def test_find_coverage_for_claim() -> None:
    coverage = CoverageRecord(
        jurisdiction="SAMPLE",
        indicator_id="6.4",
        state=CoverageState.EVIDENCE_FOUND,
        verified_claim_ids=["claim_001"],
    )
    assert find_coverage_for_claim(_claim(), [coverage]) is coverage
    assert find_coverage_for_claim(_claim(), []) is None


def test_regime_open_questions_flags_sparse_regime(
    indicator_64: IndicatorConfig,
) -> None:
    coverage = CoverageRecord(
        jurisdiction="SAMPLE",
        indicator_id="6.4",
        state=CoverageState.EVIDENCE_FOUND,
        verified_claim_ids=["claim_001"],
    )
    questions = regime_open_questions(_claim(), indicator_64, coverage)
    assert any("primary clause" in q for q in questions)
    assert any("definitional cross-references" in q for q in questions)


def test_score_from_few_shots_matches_example(indicator_64: IndicatorConfig) -> None:
    band = score_from_few_shots(_claim(), indicator_64)
    assert band is ScoreBand.HALF


def test_pick_band_prefers_few_shot() -> None:
    scores = {ScoreBand.ZERO: 10.0, ScoreBand.HALF: 0.0, ScoreBand.ONE: 0.0}
    assert pick_band(scores, ScoreBand.HALF) is ScoreBand.HALF


def test_pick_band_defaults_to_half_on_tie() -> None:
    scores = {ScoreBand.ZERO: 1.0, ScoreBand.HALF: 1.0, ScoreBand.ONE: 1.0}
    assert pick_band(scores, None) is ScoreBand.HALF


def test_build_layer2_batch_for_pillar(
    evidence_coverage: CoverageRecord,
) -> None:
    recs = build_layer2_batch(
        jurisdiction="SAMPLE",
        pillars=[_pillar_6()],
        verified_claims=[_claim()],
        coverage_records=[evidence_coverage],
    )
    assert len(recs) == 1
    assert recs[0].recommended_band is ScoreBand.HALF


def test_enrich_claim_regime_walks_graph() -> None:
    elements = {
        eid: Element(
            element_id=eid,
            doc_id="d1",
            element_type=ElementType.PARAGRAPH,
            text="x",
            page=1,
            char_start=0,
            char_end=1,
            extraction_confidence=1.0,
        )
        for eid in ("elem_1", "prov")
    }
    edges = [
        StructureEdge(
            from_element="elem_1",
            to_element="prov",
            edge_type=StructureEdgeType.NOTWITHSTANDING,
        ),
    ]
    enriched = enrich_claim_regime(_claim(), elements, edges, max_depth=1)
    assert "prov" in enriched.regime.exceptions


def test_assemble_regime_from_claim_raises_when_unresolvable() -> None:
    claim = _claim(
        regime=LegalRegime(primary_element_id="missing", member_element_ids=["missing"]),
    ).model_copy(update={"clause_id": "also_missing"})
    with pytest.raises(ValueError, match="not in graph"):
        assemble_regime_from_claim(claim, {}, [])
