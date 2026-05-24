"""Profile-name → strategy class lookup.

Add a profile = one new file in `strategies/` + one entry here. Nothing else
in the engine moves.
"""

from __future__ import annotations

from rie_contracts import DocumentProfile

from rie_profiles.base import DocumentProfileStrategy, ProfileError

_REGISTRY: dict[DocumentProfile, type[DocumentProfileStrategy]] = {}


def register_strategy(profile: DocumentProfile, cls: type[DocumentProfileStrategy]) -> None:
    if profile in _REGISTRY and _REGISTRY[profile] is not cls:
        raise ProfileError(f"duplicate profile registration: {profile}")
    _REGISTRY[profile] = cls


def get_strategy(profile: DocumentProfile) -> DocumentProfileStrategy:
    cls = _REGISTRY.get(profile)
    if cls is None:
        raise ProfileError(
            f"no strategy registered for profile {profile!r}. "
            f"Known: {sorted(p.value for p in _REGISTRY)}"
        )
    return cls()


def available_profiles() -> list[DocumentProfile]:
    return sorted(_REGISTRY)
