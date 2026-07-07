"""Gold-set confidence screening and validated-confidence routing — §6.7."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

from rie_contracts import (
    AuthorityTier,
    Claim,
    ClausePattern,
    Decomposition,
    EvidenceSpan,
    GateName,
    GateResult,
    GoldItem,
    Layer1Status,
    LegalRegime,
    ScoreBand,
    SpanRole,
    VerificationReport,
    VerificationStatus,
)
from rie_verify import (
    FakeNliBackend,
    FakeSecondLlm,
    VerificationService,
    apply_confidence_routing,
    route_verifications_by_confidence,
    screen_confidence_on_gold,
)
from rie_verify.confidence import ConfidenceScreenResult


def _claim(
    confidence: float | None,
    *,
    claim_id: str = "c-1",
    indicator: str = "6.4",
    doc_id: str = "doc1",
) -> Claim:
    return Claim(
        claim_id=claim_id,
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
        regime=LegalRegime(
            primary_element_id=f"{doc_id}_p1",
            member_element_ids=[f"{doc_id}_p1"],
        ),
        layer1_status=Layer1Status.PENDING_VERIFICATION,
        model_confidence=confidence,
        self_consistency_votes={indicator: 3},
        created_at=datetime.now(tz=UTC),
    )


def _gold(indicator: str = "6.4", doc_id: str = "doc1") -> GoldItem:
    return GoldItem(
        gold_id=f"g-{indicator}-{doc_id}",
        doc_id=doc_id,
        jurisdiction="SAMPLE",
        pillar_id="6",
        indicator_id=indicator,
        span_text="sample text",
        expected_clause_pattern=ClausePattern.CONDITIONAL_REGIME,
        expected_score_band=ScoreBand.HALF,
        expected_authority_tier=AuthorityTier.TIER_1_STATUTE,
    )


def _verified_report(claim_id: str = "c-1") -> VerificationReport:
    now = datetime.now(tz=UTC)
    gates = [GateResult(gate=gate, passed=True, detail="ok", ran_at=now) for gate in GateName]
    return VerificationReport(
        claim_id=claim_id,
        gates=gates,
        status=VerificationStatus.VERIFIED,
        failure_reasons=[],
    )


def test_screen_confidence_usable_when_separation_high() -> None:
    claims = [
        _claim(0.9, claim_id="c-1", indicator="6.4", doc_id="doc1"),
        _claim(0.4, claim_id="c-2", indicator="6.3", doc_id="doc2"),
    ]
    gold = [_gold("6.4", "doc1"), _gold("6.5", "doc2")]
    result = screen_confidence_on_gold(gold, claims, min_pairs=2, min_separation=0.1)
    assert result.usable is True
    assert result.threshold is not None


def test_screen_confidence_not_usable_with_insufficient_pairs() -> None:
    result = screen_confidence_on_gold([_gold()], [_claim(0.8)], min_pairs=3)
    assert result.usable is False
    assert result.insufficient_data is True
    assert result.screened_pairs == 1


def test_routing_when_confidence_not_usable() -> None:
    screen = ConfidenceScreenResult(
        usable=False,
        threshold=None,
        separation_score=0.02,
        screened_pairs=4,
        insufficient_data=False,
    )
    report = apply_confidence_routing(_verified_report(), _claim(0.95), screen)
    assert report.status is VerificationStatus.FLAGGED
    assert any("not screened on gold set" in reason for reason in report.failure_reasons)


def test_routing_when_insufficient_data_leaves_verified() -> None:
    screen = ConfidenceScreenResult(
        usable=False,
        threshold=None,
        separation_score=0.0,
        screened_pairs=1,
        insufficient_data=True,
    )
    report = apply_confidence_routing(_verified_report(), _claim(0.95), screen)
    assert report.status is VerificationStatus.VERIFIED


def test_routing_by_validated_threshold() -> None:
    screen = ConfidenceScreenResult(
        usable=True,
        threshold=0.65,
        separation_score=0.3,
        screened_pairs=5,
        insufficient_data=False,
    )
    low = apply_confidence_routing(_verified_report("low"), _claim(0.5, claim_id="low"), screen)
    high = apply_confidence_routing(_verified_report("high"), _claim(0.9, claim_id="high"), screen)
    assert low.status is VerificationStatus.FLAGGED
    assert any("below validated threshold" in reason for reason in low.failure_reasons)
    assert high.status is VerificationStatus.VERIFIED


def test_route_verifications_batch_applies_per_pillar_screen() -> None:
    claims = [
        _claim(0.9, claim_id="good", indicator="6.4", doc_id="doc1"),
        _claim(0.2, claim_id="bad", indicator="6.3", doc_id="doc2"),
    ]
    gold = [
        _gold("6.4", "doc1"),
        _gold("6.5", "doc2"),
        _gold("6.5", "doc3"),
        _gold("6.5", "doc4"),
    ]
    verifications = {
        "good": _verified_report("good"),
        "bad": _verified_report("bad"),
    }
    routed, screens = route_verifications_by_confidence(
        verifications,
        claims,
        {"6": gold},
        min_pairs=2,
        min_separation=0.1,
    )
    assert "6" in screens
    assert screens["6"].usable is True
    assert routed["good"].status is VerificationStatus.VERIFIED
    assert routed["bad"].status is VerificationStatus.FLAGGED


def test_verify_with_confidence_routing_preserves_four_gates(
    claim_factory: Callable[..., Claim],
    resolver: Callable[[str], str],
) -> None:
    svc = VerificationService(
        nli_model=FakeNliBackend(entailment_prob=0.9),
        second_llm=FakeSecondLlm(verdict=True),
    )
    claim = claim_factory().model_copy(update={"model_confidence": 0.2})
    screen = ConfidenceScreenResult(
        usable=True,
        threshold=0.65,
        separation_score=0.3,
        screened_pairs=5,
        insufficient_data=False,
    )
    report = svc.verify_with_confidence_routing(claim, resolver, screen)
    assert len(report.gates) == 4
    assert report.status is VerificationStatus.FLAGGED
    assert all(g.passed for g in report.gates)
