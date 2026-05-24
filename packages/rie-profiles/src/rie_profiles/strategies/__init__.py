"""Concrete profile strategies. Importing this module wires the registry."""

from rie_profiles.registry import register_strategy
from rie_profiles.strategies.mixed_regulatory import MixedRegulatoryStrategy
from rie_profiles.strategies.statutory_legal_text import StatutoryLegalTextStrategy
from rie_profiles.strategies.structured_tabular import StructuredTabularStrategy
from rie_profiles.strategies.treaty_membership import TreatyMembershipStrategy

for _cls in (
    StatutoryLegalTextStrategy,
    StructuredTabularStrategy,
    MixedRegulatoryStrategy,
    TreatyMembershipStrategy,
):
    register_strategy(_cls.profile, _cls)

__all__ = [
    "MixedRegulatoryStrategy",
    "StatutoryLegalTextStrategy",
    "StructuredTabularStrategy",
    "TreatyMembershipStrategy",
]
