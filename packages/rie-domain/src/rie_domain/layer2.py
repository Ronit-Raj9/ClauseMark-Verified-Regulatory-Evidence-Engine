"""Deterministic Layer-2 score-band recommendation (§6.4).

The LLM never authors a score. This module maps verified Layer-1 claims,
indicator ``scoring_criteria``, and assembled regime metadata to a
:class:`~rie_contracts.Layer2Recommendation`. ``human_confirmation_required``
is always ``True``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from rie_contracts import (
    Claim,
    ClausePattern,
    CoverageRecord,
    CoverageState,
    IndicatorConfig,
    Layer2Recommendation,
    PillarConfig,
    ScoreBand,
)

_BAND_ORDER: tuple[ScoreBand, ...] = (ScoreBand.ZERO, ScoreBand.HALF, ScoreBand.ONE)
_NUMERIC_BANDS: frozenset[ScoreBand] = frozenset(_BAND_ORDER)
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def claim_text(claim: Claim) -> str:
    """Flatten decomposition + clause pattern into searchable text."""
    parts = [
        claim.clause_pattern.value,
        claim.decomposition.subject,
        claim.decomposition.condition or "",
        claim.decomposition.constraint,
        claim.decomposition.context or "",
    ]
    return " ".join(p for p in parts if p)


def _keywords_for_indicator(indicator: IndicatorConfig) -> tuple[list[str], list[str]]:
    positive: list[str] = []
    negative: list[str] = []
    for values in indicator.positive_keywords.values():
        positive.extend(values)
    for values in indicator.negative_cues.values():
        negative.extend(values)
    return positive, negative


def _band_from_numeric(value: float | str) -> ScoreBand | None:
    if isinstance(value, str):
        mapping = {
            "0": ScoreBand.ZERO,
            "0.5": ScoreBand.HALF,
            "1": ScoreBand.ONE,
            "null": ScoreBand.NULL,
        }
        return mapping.get(value)
    mapping = {0.0: ScoreBand.ZERO, 0.5: ScoreBand.HALF, 1.0: ScoreBand.ONE}
    return mapping.get(float(value))


def score_bands_from_criteria(
    claim: Claim,
    indicator: IndicatorConfig,
) -> dict[ScoreBand, float]:
    """Score each rubric band by keyword / criteria overlap with the claim."""
    text = claim_text(claim).lower()
    tokens = _tokenize(text)
    positive, negative = _keywords_for_indicator(indicator)
    scores: dict[ScoreBand, float] = dict.fromkeys(_BAND_ORDER, 0.0)

    for band_key, description in indicator.scoring_criteria.items():
        band = _band_from_numeric(band_key)
        if band is None or band not in _NUMERIC_BANDS:
            continue
        desc_tokens = _tokenize(description)
        overlap = len(tokens & desc_tokens)
        scores[band] += float(overlap)

    for kw in positive:
        if kw.lower() in text:
            for band in _BAND_ORDER:
                scores[band] += 0.25

    for cue in negative:
        if cue.lower() in text:
            scores[ScoreBand.ONE] -= 0.5
            scores[ScoreBand.HALF] += 0.25

    if indicator.clause_pattern is not None and claim.clause_pattern == indicator.clause_pattern:
        scores[ScoreBand.HALF] += 0.5

    if claim.clause_pattern is ClausePattern.CONDITIONAL_REGIME:
        scores[ScoreBand.HALF] += 0.75
        scores[ScoreBand.ONE] -= 0.25

    return scores


def score_from_few_shots(
    claim: Claim,
    indicator: IndicatorConfig,
) -> ScoreBand | None:
    """Pick the band from the closest config few-shot example, if any."""
    claim_tokens = _tokenize(claim_text(claim))
    if not claim_tokens or not indicator.few_shot_examples:
        return None

    best_band: ScoreBand | None = None
    best_score = 0.0

    for example in indicator.few_shot_examples:
        example_label = str(example.get("label", ""))
        if example_label and example_label != indicator.indicator_id:
            continue
        score_val = example.get("score")
        band = _band_from_numeric(score_val) if score_val is not None else None
        if band is None or band not in _NUMERIC_BANDS:
            continue

        example_parts: list[str] = []
        if isinstance(example.get("text"), str):
            example_parts.append(example["text"])
        decomp = example.get("decomposition")
        if isinstance(decomp, dict):
            example_parts.extend(str(v) for v in decomp.values() if v)

        example_tokens = _tokenize(" ".join(example_parts))
        if not example_tokens:
            continue

        overlap = len(claim_tokens & example_tokens) / len(claim_tokens | example_tokens)
        if overlap > best_score:
            best_score = overlap
            best_band = band

    if best_score >= 0.15:
        return best_band
    return None


def regime_open_questions(
    claim: Claim,
    indicator: IndicatorConfig,
    coverage: CoverageRecord,
) -> list[str]:
    """Reviewer prompts derived from regime assembly + coverage gaps."""
    questions: list[str] = []

    if coverage.state is CoverageState.INSUFFICIENT_COVERAGE:
        questions.append(
            "Corpus coverage is insufficient — widen authoritative sources before confirming a score."
        )
    elif coverage.state is CoverageState.NO_EVIDENCE_IN_SEARCHED_CORPUS:
        questions.append(
            "No verified evidence for this indicator in the searched corpus — "
            "confirm whether a RDTII score is defensible."
        )

    regime = claim.regime
    if len(regime.member_element_ids) <= 1:
        questions.append(
            "Regime graph contains only the primary clause — "
            "confirm whether implementing regulations or exceptions exist elsewhere."
        )
    if not regime.exceptions and claim.clause_pattern is ClausePattern.CONDITIONAL_REGIME:
        questions.append(
            "Conditional-regime structure detected but no exception members were assembled — "
            "confirm adequacy-list breadth and authorisation mechanics."
        )
    if not regime.definitions:
        questions.append(
            "No definitional cross-references were assembled — "
            "confirm scope terms (e.g. personal data categories) from other sections."
        )
    if indicator.clause_pattern is not None and claim.clause_pattern != indicator.clause_pattern:
        questions.append(
            f"Claim clause pattern ({claim.clause_pattern.value}) differs from the "
            f"indicator's expected pattern ({indicator.clause_pattern.value}) — "
            "confirm the mapping before scoring."
        )

    return questions


def pick_band(scores: dict[ScoreBand, float], few_shot_band: ScoreBand | None) -> ScoreBand:
    """Choose a band; prefer few-shot when present, else highest criteria score."""
    if few_shot_band is not None:
        return few_shot_band

    ranked = sorted(_BAND_ORDER, key=lambda b: scores.get(b, 0.0), reverse=True)
    top, second = ranked[0], ranked[1]
    top_score = scores.get(top, 0.0)
    second_score = scores.get(second, 0.0)

    if top_score <= 0.0:
        return ScoreBand.HALF

    if top_score - second_score < 0.5:
        return ScoreBand.HALF

    return top


def format_rationale(
    claim: Claim,
    indicator: IndicatorConfig,
    band: ScoreBand,
    coverage: CoverageRecord,
) -> str:
    """Human-readable rationale citing config rubric + regime context."""
    rubric = indicator.scoring_criteria.get(band.value, "")
    member_count = len(claim.regime.member_element_ids)
    exception_count = len(claim.regime.exceptions)
    definition_count = len(claim.regime.definitions)

    parts = [
        f"Structure = {indicator.name} ({claim.clause_pattern.value}); "
        f"recommended band {band.value}.",
    ]
    if rubric:
        parts.append(f"Rubric: {rubric}")
    parts.append(
        f"Regime assembly: {member_count} member(s), "
        f"{exception_count} exception(s), {definition_count} definition(s)."
    )
    if coverage.state is CoverageState.EVIDENCE_FOUND:
        parts.append("Coverage state: evidence_found.")
    else:
        parts.append(f"Coverage state: {coverage.state.value}.")
    parts.append("Final restrictiveness is a regime-level judgment — reviewer must confirm.")
    return " ".join(parts)


def recommend_band(
    claim: Claim,
    indicator: IndicatorConfig,
    coverage: CoverageRecord,
) -> tuple[ScoreBand, str, list[str]]:
    """Return ``(band, rationale, open_questions)`` for one verified claim."""
    if coverage.state is not CoverageState.EVIDENCE_FOUND:
        band = ScoreBand.NULL
        rationale = (
            f"No verified evidence band recommended — coverage is {coverage.state.value}"
            + (f" ({coverage.reason})" if coverage.reason else "")
            + "."
        )
        return band, rationale, regime_open_questions(claim, indicator, coverage)

    if claim.claim_id not in coverage.verified_claim_ids:
        band = ScoreBand.NULL
        rationale = (
            "Claim is verified but not counted in coverage for this indicator — "
            "review jurisdiction / indicator alignment before scoring."
        )
        return band, rationale, regime_open_questions(claim, indicator, coverage)

    criteria_scores = score_bands_from_criteria(claim, indicator)
    few_shot_band = score_from_few_shots(claim, indicator)
    band = pick_band(criteria_scores, few_shot_band)
    open_questions = regime_open_questions(claim, indicator, coverage)
    rationale = format_rationale(claim, indicator, band, coverage)
    return band, rationale, open_questions


def build_layer2_recommendation(
    claim: Claim,
    indicator: IndicatorConfig,
    coverage: CoverageRecord,
) -> Layer2Recommendation:
    """Produce a Layer-2 recommendation from a verified claim + indicator config."""
    band, rationale, open_questions = recommend_band(claim, indicator, coverage)
    return Layer2Recommendation(
        claim_id=claim.claim_id,
        indicator_id=claim.indicator_id,
        recommended_band=band,
        rationale=rationale,
        open_questions=open_questions,
        human_confirmation_required=True,
    )


def build_layer2_from_pillar(
    claim: Claim,
    pillar: PillarConfig,
    coverage: CoverageRecord,
) -> Layer2Recommendation | None:
    """Resolve indicator from pillar config, then build Layer-2 recommendation."""
    indicator = next((i for i in pillar.indicators if i.indicator_id == claim.indicator_id), None)
    if indicator is None:
        return None
    return build_layer2_recommendation(claim, indicator, coverage)


def find_coverage_for_claim(
    claim: Claim,
    coverage_records: Sequence[CoverageRecord],
) -> CoverageRecord | None:
    """Locate the coverage row for a claim's ``(jurisdiction, indicator_id)``."""
    for record in coverage_records:
        if record.jurisdiction == claim.jurisdiction and record.indicator_id == claim.indicator_id:
            return record
    return None


def build_layer2_batch(
    *,
    jurisdiction: str,
    pillars: Sequence[PillarConfig],
    verified_claims: Sequence[Claim],
    coverage_records: Sequence[CoverageRecord],
) -> list[Layer2Recommendation]:
    """Batch Layer-2 recommendations for every ``(jurisdiction, indicator)`` in *pillars*."""
    coverage_by_key = {
        (record.jurisdiction, record.indicator_id): record for record in coverage_records
    }
    recommendations: list[Layer2Recommendation] = []

    for pillar in pillars:
        for indicator in pillar.indicators:
            key = (jurisdiction, indicator.indicator_id)
            coverage = coverage_by_key.get(key)
            if coverage is None:
                continue
            for claim in verified_claims:
                if claim.jurisdiction != jurisdiction:
                    continue
                if claim.indicator_id != indicator.indicator_id:
                    continue
                recommendations.append(
                    build_layer2_recommendation(claim, indicator, coverage)
                )
    return recommendations
