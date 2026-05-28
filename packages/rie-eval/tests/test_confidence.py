"""Confidence screening on gold set — §6.7."""

from __future__ import annotations

from datetime import UTC, datetime

from rie_contracts import (
    AuthorityTier,
    Claim,
    ClausePattern,
    Decomposition,
    EvidenceSpan,
    GoldItem,
    Layer1Status,
    LegalRegime,
    ScoreBand,
    SpanRole,
)
from rie_verify.confidence import screen_confidence_on_gold


def _claim(confidence: float, indicator: str = "6.4", doc_id: str = "doc1") -> Claim:
    return Claim(
        claim_id=f"c-{doc_id}-{confidence}",
        indicator_id=indicator,
        pillar_id="6",
        clause_id=f"{doc_id}_p1",
        jurisdiction="SAMPLE",
        clause_pattern=ClausePattern.CONDITIONAL_REGIME,
        decomposition=Decomposition(subject="org", constraint="none"),
        evidence_spans=[
            EvidenceSpan(
                span_id=f"{doc_id}#0-10",
                element_id=f"{doc_id}_p1",
                doc_id=doc_id,
                char_start=0,
                char_end=10,
                role=SpanRole.PRIMARY,
            )
        ],
        regime=LegalRegime(primary_element_id=f"{doc_id}_p1", member_element_ids=[f"{doc_id}_p1"]),
        layer1_status=Layer1Status.PENDING_VERIFICATION,
        model_confidence=confidence,
        created_at=datetime.now(tz=UTC),
    )


def _gold(indicator: str = "6.4", doc_id: str = "doc1") -> GoldItem:
    return GoldItem(
        gold_id=f"g-{doc_id}",
        doc_id=doc_id,
        jurisdiction="SAMPLE",
        pillar_id="6",
        indicator_id=indicator,
        span_text="sample text",
        expected_clause_pattern=ClausePattern.CONDITIONAL_REGIME,
        expected_score_band=ScoreBand.HALF,
        expected_authority_tier=AuthorityTier.TIER_1_STATUTE,
    )


def test_confidence_usable_when_separation_high() -> None:
    claims = [
        _claim(0.92, indicator="6.4", doc_id="doc1"),
        _claim(0.35, indicator="6.4", doc_id="doc2"),  # wrong indicator vs gold 6.5
    ]
    gold = [_gold("6.4", doc_id="doc1"), _gold("6.5", doc_id="doc2")]
    result = screen_confidence_on_gold(gold, claims, min_pairs=2, min_separation=0.1)
    assert result.usable is True
    assert result.threshold is not None


def test_confidence_not_usable_with_insufficient_pairs() -> None:
    result = screen_confidence_on_gold([_gold()], [_claim(0.8)], min_pairs=3)
    assert result.usable is False
    assert result.screened_pairs == 1
