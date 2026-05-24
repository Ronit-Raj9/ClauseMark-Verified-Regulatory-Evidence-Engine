"""Document-profile strategy registry — the all-pillar seam."""

from rie_profiles.base import DocumentProfileStrategy, ProfileError
from rie_profiles.registry import (
    available_profiles,
    get_strategy,
    register_strategy,
)
from rie_profiles.strategies import (
    MixedRegulatoryStrategy,
    StatutoryLegalTextStrategy,
    StructuredTabularStrategy,
    TreatyMembershipStrategy,
)

__all__ = [
    "DocumentProfileStrategy",
    "MixedRegulatoryStrategy",
    "ProfileError",
    "StatutoryLegalTextStrategy",
    "StructuredTabularStrategy",
    "TreatyMembershipStrategy",
    "available_profiles",
    "get_strategy",
    "register_strategy",
]
