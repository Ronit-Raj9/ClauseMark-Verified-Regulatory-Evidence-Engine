"""ADVISORY entity-grounding must NEVER affect VerificationReport.status.

A claim whose 4 required gates all pass stays VERIFIED even when the
entity-grounding advisory check FAILS. The advisory result lands in
``advisory_checks`` — the required-4 are the sole status determinant.
"""

from __future__ import annotations

from collections.abc import Callable

from rie_contracts import Claim, GateName, VerificationStatus
from rie_verify import (
    EntityGroundingChecker,
    FakeEntityExtractor,
    FakeNliBackend,
    FakeSecondLlm,
    GroundingEntity,
    VerificationService,
)


def test_advisory_failure_does_not_block_verified(
    claim_factory: Callable[..., Claim],
    resolver: Callable[[str], str],
) -> None:
    # Entity-grounding will fail: none of these occur in the cited text.
    failing_grounder = EntityGroundingChecker(
        FakeEntityExtractor(
            [GroundingEntity(text="Nowhere Ltd", label="ORG", char_start=0, char_end=11)]
        )
    )
    svc = VerificationService(
        nli_model=FakeNliBackend(entailment_prob=0.9, contradiction_prob=0.05, neutral_prob=0.05),
        second_llm=FakeSecondLlm(verdict=True),
        entity_grounder=failing_grounder,
    )

    report = svc.verify(claim_factory(), resolver)

    # Required-4 all pass → VERIFIED, regardless of advisory failure.
    assert report.status is VerificationStatus.VERIFIED
    assert len(report.gates) == 4
    assert all(g.passed for g in report.gates)

    # Advisory channel populated, the entry is not-passed, and it is NOT a gate.
    assert len(report.advisory_checks) == 1
    advisory = report.advisory_checks[0]
    assert advisory.gate is GateName.ENTITY_GROUNDING
    assert advisory.passed is False
    assert advisory.gate not in {g.gate for g in report.gates}


def test_no_grounder_means_empty_advisory(
    claim_factory: Callable[..., Claim],
    resolver: Callable[[str], str],
) -> None:
    svc = VerificationService(
        nli_model=FakeNliBackend(entailment_prob=0.9, contradiction_prob=0.05, neutral_prob=0.05),
        second_llm=FakeSecondLlm(verdict=True),
    )
    report = svc.verify(claim_factory(), resolver)

    assert report.status is VerificationStatus.VERIFIED
    assert report.advisory_checks == []
    assert svc.entity_grounding_enabled is False
