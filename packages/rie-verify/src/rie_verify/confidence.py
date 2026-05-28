"""Gold-set confidence screening — §6.7 selective prediction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from rie_contracts import Claim, GoldItem, VerificationReport, VerificationStatus


@dataclass(frozen=True)
class ConfidenceScreenResult:
    """Outcome of screening classifier confidence on the gold set."""

    usable: bool
    threshold: float | None
    separation_score: float
    screened_pairs: int
    insufficient_data: bool


def screen_confidence_on_gold(
    gold_items: Sequence[GoldItem],
    claims: Sequence[Claim],
    *,
    min_pairs: int = 3,
    min_separation: float = 0.15,
) -> ConfidenceScreenResult:
    """Screen classifier confidence against gold-set correctness."""
    pairs: list[tuple[float, bool]] = []

    for item in gold_items:
        claim = _match_gold_to_claim(item, claims)
        if claim is None or claim.model_confidence is None:
            continue
        correct = claim.indicator_id == item.indicator_id
        pairs.append((claim.model_confidence, correct))

    if len(pairs) < min_pairs:
        return ConfidenceScreenResult(
            usable=False,
            threshold=None,
            separation_score=0.0,
            screened_pairs=len(pairs),
            insufficient_data=True,
        )

    correct_conf = [c for c, ok in pairs if ok]
    incorrect_conf = [c for c, ok in pairs if not ok]
    if not correct_conf or not incorrect_conf:
        return ConfidenceScreenResult(
            usable=False,
            threshold=None,
            separation_score=0.0,
            screened_pairs=len(pairs),
            insufficient_data=False,
        )

    mean_correct = sum(correct_conf) / len(correct_conf)
    mean_incorrect = sum(incorrect_conf) / len(incorrect_conf)
    separation = mean_correct - mean_incorrect
    usable = separation >= min_separation
    threshold = (mean_correct + mean_incorrect) / 2.0 if usable else None

    return ConfidenceScreenResult(
        usable=usable,
        threshold=threshold,
        separation_score=max(0.0, min(1.0, separation)),
        screened_pairs=len(pairs),
        insufficient_data=False,
    )


def _match_gold_to_claim(item: GoldItem, claims: Sequence[Claim]) -> Claim | None:
    candidates: list[Claim] = []
    for claim in claims:
        if claim.jurisdiction != item.jurisdiction:
            continue
        if any(sp.doc_id == item.doc_id for sp in claim.evidence_spans):
            candidates.append(claim)
    if not candidates:
        return None
    for claim in candidates:
        if claim.indicator_id == item.indicator_id:
            return claim
    return candidates[0]


def confidence_routing_reason(
    claim: Claim,
    screen: ConfidenceScreenResult,
) -> str | None:
    """Return a review-routing reason, or ``None`` when confidence should not reroute."""
    if screen.insufficient_data:
        return None
    if not screen.usable:
        return "confidence not screened on gold set — routed to review"
    if screen.threshold is None:
        return None
    if claim.model_confidence is None:
        return "model confidence missing — routed to review"
    if claim.model_confidence < screen.threshold:
        return (
            f"model confidence {claim.model_confidence:.3f} below validated "
            f"threshold {screen.threshold:.3f}"
        )
    return None


def apply_confidence_routing(
    report: VerificationReport,
    claim: Claim,
    screen: ConfidenceScreenResult,
) -> VerificationReport:
    """Apply §6.7 routing to a single verification report (post 4-gate ``verify``)."""
    if report.status is not VerificationStatus.VERIFIED:
        return report
    reason = confidence_routing_reason(claim, screen)
    if reason is None:
        return report
    return report.model_copy(
        update={
            "status": VerificationStatus.FLAGGED,
            "failure_reasons": [*report.failure_reasons, reason],
        }
    )


def route_verifications_by_confidence(
    verifications: Mapping[str, VerificationReport],
    claims: Sequence[Claim],
    gold_by_pillar: Mapping[str, Sequence[GoldItem]],
    *,
    min_pairs: int = 3,
    min_separation: float = 0.15,
) -> tuple[dict[str, VerificationReport], dict[str, ConfidenceScreenResult]]:
    """Screen classifier confidence per pillar and route VERIFIED claims to review."""
    claims_by_id = {claim.claim_id: claim for claim in claims}
    screens: dict[str, ConfidenceScreenResult] = {}
    for pillar_id, gold_items in gold_by_pillar.items():
        pillar_claims = [claim for claim in claims if claim.pillar_id == pillar_id]
        screens[pillar_id] = screen_confidence_on_gold(
            gold_items,
            pillar_claims,
            min_pairs=min_pairs,
            min_separation=min_separation,
        )

    updated = dict(verifications)
    for claim_id, report in verifications.items():
        claim = claims_by_id.get(claim_id)
        if claim is None:
            continue
        screen = screens.get(claim.pillar_id)
        if screen is None:
            continue
        routed = apply_confidence_routing(report, claim, screen)
        if routed is not report:
            updated[claim_id] = routed
    return updated, screens
