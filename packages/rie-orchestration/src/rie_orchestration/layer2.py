"""Orchestration helper — resolve Layer-2 recommendations for API / graph nodes."""

from __future__ import annotations

from collections.abc import Sequence

from rie_config import ConfigRepository
from rie_contracts import (
    Claim,
    ConfigRepositoryPort,
    CoverageRecord,
    Layer2Recommendation,
)
from rie_coverage import Layer2ScoringService

__all__ = ["materialise_layer2_batch", "materialise_layer2_for_claim"]


def materialise_layer2_for_claim(
    claim: Claim,
    *,
    coverage_records: Sequence[CoverageRecord],
    config: ConfigRepositoryPort,
    scorer: Layer2ScoringService | None = None,
) -> Layer2Recommendation | None:
    """Compute the Layer-2 recommendation for one claim, if config + coverage exist."""
    svc = scorer or Layer2ScoringService()
    pillar = config.load_pillar(claim.pillar_id)
    indicator = next((i for i in pillar.indicators if i.indicator_id == claim.indicator_id), None)
    if indicator is None:
        return None
    coverage = next(
        (
            c
            for c in coverage_records
            if c.jurisdiction == claim.jurisdiction and c.indicator_id == claim.indicator_id
        ),
        None,
    )
    if coverage is None:
        return None
    return svc.recommend_for_claim(claim, indicator, coverage)


def materialise_layer2_batch(
    *,
    jurisdiction: str,
    pillar_ids: Sequence[str],
    verified_claims: Sequence[Claim],
    coverage_records: Sequence[CoverageRecord],
    config: ConfigRepository | ConfigRepositoryPort,
    scorer: Layer2ScoringService | None = None,
) -> list[Layer2Recommendation]:
    """Batch Layer-2 recommendations for a pipeline run."""
    svc = scorer or Layer2ScoringService()
    pillars = [config.load_pillar(pid) for pid in pillar_ids]
    return svc.recommend_all(
        jurisdiction=jurisdiction,
        pillars=pillars,
        verified_claims=verified_claims,
        coverage_records=coverage_records,
    )
