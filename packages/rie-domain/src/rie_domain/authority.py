"""Authority-tier resolution with jurisdiction-specific overrides."""

from __future__ import annotations

from dataclasses import dataclass, field

from rie_contracts import AuthorityTier

_RANK: dict[AuthorityTier, int] = {
    AuthorityTier.TIER_1_STATUTE: 4,
    AuthorityTier.TIER_2_REGULATION: 3,
    AuthorityTier.TIER_3_GUIDELINE: 2,
    AuthorityTier.TIER_4_INFORMAL: 1,
}


def rank_authority(tier: AuthorityTier) -> int:
    return _RANK[tier]


@dataclass(frozen=True)
class AuthorityOverride:
    jurisdiction: str
    source_pattern: str  # regex over source_url or source_id
    tier: AuthorityTier
    rationale: str


@dataclass
class AuthorityResolver:
    """Applies global defaults then jurisdiction-specific overrides.

    The default tier comes from the source registry's declaration; an override
    can elevate (e.g. a regulator-issued circular treated as binding in a
    civil-law jurisdiction) or demote.
    """

    overrides: list[AuthorityOverride] = field(default_factory=list)

    def resolve(
        self,
        jurisdiction: str,
        source_id: str,
        source_url: str | None,
        declared_tier: AuthorityTier,
    ) -> AuthorityTier:
        import re

        for ov in self.overrides:
            if ov.jurisdiction != jurisdiction:
                continue
            haystacks = [source_id]
            if source_url:
                haystacks.append(source_url)
            if any(re.search(ov.source_pattern, h) for h in haystacks):
                return ov.tier
        return declared_tier


def apply_authority_overrides(
    declared: AuthorityTier,
    overrides: list[AuthorityOverride],
    jurisdiction: str,
    source_id: str,
    source_url: str | None = None,
) -> AuthorityTier:
    return AuthorityResolver(overrides=overrides).resolve(
        jurisdiction, source_id, source_url, declared
    )
