"""ADVISORY entity-grounding gate — unit tests using FakeEntityExtractor.

These never require a spaCy install. They assert score/passed semantics and
that the emitted GateResult carries gate == GateName.ENTITY_GROUNDING.
"""

from __future__ import annotations

from collections.abc import Callable

from rie_contracts import Claim, GateName
from rie_verify.entity_grounding import (
    Entity as GroundingEntity,
)
from rie_verify.entity_grounding import (
    EntityGroundingChecker,
    FakeEntityExtractor,
)

# `GroundingEntity` is the entity_grounding.Entity (text, label, start, end).


def test_all_entities_grounded_scores_one_and_passes(
    claim_factory: Callable[..., Claim],
    resolver: Callable[[str], str],
) -> None:
    # All decomposition entities appear verbatim in the cited/regime text.
    canned = [
        GroundingEntity(text="organisation", label="ORG", char_start=0, char_end=12),
        GroundingEntity(text="personal data", label="LAW", char_start=0, char_end=13),
        GroundingEntity(text="protection", label="LAW", char_start=0, char_end=10),
    ]
    checker = EntityGroundingChecker(FakeEntityExtractor(canned))
    claim = claim_factory()

    result = checker.check(claim, resolver)

    assert result.gate is GateName.ENTITY_GROUNDING
    assert result.score == 1.0
    assert result.passed is True
    assert "3/3" in result.detail
    assert "advisory only" in result.detail


def test_no_entities_grounded_scores_zero_and_not_passed(
    claim_factory: Callable[..., Claim],
    resolver: Callable[[str], str],
) -> None:
    # None of these strings occur in the cited/regime text.
    canned = [
        GroundingEntity(text="Acme Corporation", label="ORG", char_start=0, char_end=16),
        GroundingEntity(text="Antarctica", label="GPE", char_start=0, char_end=10),
    ]
    checker = EntityGroundingChecker(FakeEntityExtractor(canned))
    claim = claim_factory()

    result = checker.check(claim, resolver)

    assert result.gate is GateName.ENTITY_GROUNDING
    assert result.score == 0.0
    assert result.passed is False
    assert "0/2" in result.detail


def test_partial_grounding_respects_threshold(
    claim_factory: Callable[..., Claim],
    resolver: Callable[[str], str],
) -> None:
    # One of two grounded → score 0.5; default threshold 0.5 → passed.
    canned = [
        GroundingEntity(text="organisation", label="ORG", char_start=0, char_end=12),
        GroundingEntity(text="Atlantis", label="GPE", char_start=0, char_end=8),
    ]
    checker = EntityGroundingChecker(FakeEntityExtractor(canned))
    claim = claim_factory()

    result = checker.check(claim, resolver)

    assert result.gate is GateName.ENTITY_GROUNDING
    assert result.score == 0.5
    assert result.passed is True  # 0.5 >= 0.5

    strict = EntityGroundingChecker(FakeEntityExtractor(canned), threshold=0.6)
    strict_result = strict.check(claim, resolver)
    assert strict_result.passed is False


def test_no_named_entities_is_vacuous_pass(
    claim_factory: Callable[..., Claim],
    resolver: Callable[[str], str],
) -> None:
    checker = EntityGroundingChecker(FakeEntityExtractor([]))
    claim = claim_factory()

    result = checker.check(claim, resolver)

    assert result.gate is GateName.ENTITY_GROUNDING
    assert result.passed is True
    assert result.score == 1.0
    assert "0/0" in result.detail
