"""Orchestration hook — gold-set confidence screening before routing."""

from __future__ import annotations

from collections.abc import Sequence

from rie_config import ConfigRepository
from rie_contracts import Claim
from rie_verify.confidence import ConfidenceScreenResult, screen_confidence_on_gold

__all__ = ["screen_pillar_confidence"]


def screen_pillar_confidence(
    pillar_id: str,
    claims: Sequence[Claim],
    config: ConfigRepository,
) -> ConfidenceScreenResult:
    gold = list(config.load_gold(pillar_id))
    pillar_claims = [c for c in claims if c.pillar_id == pillar_id]
    return screen_confidence_on_gold(gold, pillar_claims)
