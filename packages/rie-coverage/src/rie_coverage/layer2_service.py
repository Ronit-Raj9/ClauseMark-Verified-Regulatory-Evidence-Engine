"""Layer-2 scoring service — deterministic score-band recommendations.

Integration points
--------------------
* **Orchestration** — ``layer2_node`` in ``rie_orchestration.nodes`` calls
  :meth:`Layer2ScoringService.recommend_all` after ``coverage_node`` and stores
  results in ``RieState["layer2_recommendations"]``.
* **API** — ``GET /v1/claims/{id}`` recomputes via
  ``rie_orchestration.layer2.materialise_layer2_for_claim`` (same algorithm,
  no persistence table yet).
* **UI** — Streamlit / web clients read ``layer2`` from claim detail; reviewers
  submit the authoritative score via ``POST /v1/reviews``.
"""

from __future__ import annotations

from collections.abc import Sequence

from rie_contracts import (
    Claim,
    CoverageRecord,
    IndicatorConfig,
    Layer2Recommendation,
    PillarConfig,
)

from rie_coverage.scoring_policy import build_layer2_recommendation


class Layer2ScoringService:
    """Produce :class:`~rie_contracts.Layer2Recommendation` from verified claims.

    Pure and deterministic: no LLM calls, no I/O. The orchestrator invokes
    this after coverage evaluation; the API may recompute on read using the
    same service for consistency.
    """

    def recommend_for_claim(
        self,
        claim: Claim,
        indicator: IndicatorConfig,
        coverage: CoverageRecord,
    ) -> Layer2Recommendation:
        """Build one Layer-2 recommendation for a single verified claim."""
        return build_layer2_recommendation(claim, indicator, coverage)

    def recommend_for_indicator(
        self,
        *,
        jurisdiction: str,
        indicator: IndicatorConfig,
        verified_claims: Sequence[Claim],
        coverage: CoverageRecord,
    ) -> list[Layer2Recommendation]:
        """Recommend for every verified claim matching ``(jurisdiction, indicator)``."""
        out: list[Layer2Recommendation] = []
        for claim in verified_claims:
            if claim.jurisdiction != jurisdiction:
                continue
            if claim.indicator_id != indicator.indicator_id:
                continue
            out.append(self.recommend_for_claim(claim, indicator, coverage))
        return out

    def recommend_all(
        self,
        *,
        jurisdiction: str,
        pillars: Sequence[PillarConfig],
        verified_claims: Sequence[Claim],
        coverage_records: Sequence[CoverageRecord],
    ) -> list[Layer2Recommendation]:
        """Recommend for every ``(jurisdiction, indicator)`` in the supplied pillars."""
        recommendations: list[Layer2Recommendation] = []
        coverage_by_key = {
            (record.jurisdiction, record.indicator_id): record for record in coverage_records
        }

        for pillar in pillars:
            for indicator in pillar.indicators:
                key = (jurisdiction, indicator.indicator_id)
                coverage = coverage_by_key.get(key)
                if coverage is None:
                    continue
                recommendations.extend(
                    self.recommend_for_indicator(
                        jurisdiction=jurisdiction,
                        indicator=indicator,
                        verified_claims=verified_claims,
                        coverage=coverage,
                    )
                )
        return recommendations
