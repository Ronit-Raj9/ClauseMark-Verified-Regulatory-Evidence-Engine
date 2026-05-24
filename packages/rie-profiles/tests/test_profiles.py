import pytest
from rie_contracts import DocumentProfile
from rie_profiles import (
    DocumentProfileStrategy,
    ProfileError,
    available_profiles,
    get_strategy,
)


def test_all_four_profiles_registered() -> None:
    assert set(available_profiles()) == set(DocumentProfile)


def test_strategy_implements_protocol() -> None:
    s = get_strategy(DocumentProfile.STATUTORY_LEGAL_TEXT)
    assert isinstance(s, DocumentProfileStrategy)


def test_unknown_profile_raises() -> None:
    from rie_profiles.registry import _REGISTRY

    saved = dict(_REGISTRY)
    _REGISTRY.pop(DocumentProfile.TREATY_MEMBERSHIP, None)
    try:
        with pytest.raises(ProfileError):
            get_strategy(DocumentProfile.TREATY_MEMBERSHIP)
    finally:
        _REGISTRY.clear()
        _REGISTRY.update(saved)


def test_statutory_legal_chunk_sizes() -> None:
    s = get_strategy(DocumentProfile.STATUTORY_LEGAL_TEXT)
    assert s.child_chunk_size() > 0
    assert s.supports_llm_classification() is True


def test_treaty_membership_skips_llm() -> None:
    s = get_strategy(DocumentProfile.TREATY_MEMBERSHIP)
    assert s.supports_llm_classification() is False
