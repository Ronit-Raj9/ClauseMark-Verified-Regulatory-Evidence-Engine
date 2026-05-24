"""Gate 3 — entailment. NLI + second LLM; any disagreement → flagged."""

from __future__ import annotations

from collections.abc import Callable

from rie_contracts import Claim, GateName
from rie_verify import FakeNliBackend, FakeSecondLlm, VerificationService


def test_entailment_passes_when_both_backends_agree(
    claim_factory: Callable[..., Claim],
    resolver: Callable[[str], str],
) -> None:
    svc = VerificationService(
        nli_model=FakeNliBackend(entailment_prob=0.9, contradiction_prob=0.05, neutral_prob=0.05),
        second_llm=FakeSecondLlm(verdict=True),
    )
    claim = claim_factory()
    result = svc.run_gate(GateName.ENTAILMENT, claim, resolver)
    assert result.passed is True
    assert result.score is not None and result.score > 0.6


def test_entailment_flagged_when_nli_says_entail_but_llm_disagrees(
    claim_factory: Callable[..., Claim],
    resolver: Callable[[str], str],
) -> None:
    svc = VerificationService(
        nli_model=FakeNliBackend(entailment_prob=0.95, contradiction_prob=0.02, neutral_prob=0.03),
        second_llm=FakeSecondLlm(verdict=False),
    )
    claim = claim_factory()
    result = svc.run_gate(GateName.ENTAILMENT, claim, resolver)
    assert result.passed is False
    assert "NLI/LLM disagree" in result.detail


def test_entailment_flagged_when_llm_yes_but_nli_low(
    claim_factory: Callable[..., Claim],
    resolver: Callable[[str], str],
) -> None:
    svc = VerificationService(
        nli_model=FakeNliBackend(entailment_prob=0.30, contradiction_prob=0.20, neutral_prob=0.50),
        second_llm=FakeSecondLlm(verdict=True),
    )
    claim = claim_factory()
    result = svc.run_gate(GateName.ENTAILMENT, claim, resolver)
    assert result.passed is False
    assert "NLI/LLM disagree" in result.detail


def test_entailment_fails_when_both_reject(
    claim_factory: Callable[..., Claim],
    resolver: Callable[[str], str],
) -> None:
    svc = VerificationService(
        nli_model=FakeNliBackend(entailment_prob=0.10, contradiction_prob=0.80, neutral_prob=0.10),
        second_llm=FakeSecondLlm(verdict=False),
    )
    claim = claim_factory()
    result = svc.run_gate(GateName.ENTAILMENT, claim, resolver)
    assert result.passed is False
    assert "both verifiers reject" in result.detail


def test_entailment_threshold_is_strictly_greater_than(
    claim_factory: Callable[..., Claim],
    resolver: Callable[[str], str],
) -> None:
    """Exactly 0.6 must NOT pass — gate requires strict `>`."""
    svc = VerificationService(
        nli_model=FakeNliBackend(entailment_prob=0.6, contradiction_prob=0.2, neutral_prob=0.2),
        second_llm=FakeSecondLlm(verdict=True),
    )
    claim = claim_factory()
    result = svc.run_gate(GateName.ENTAILMENT, claim, resolver)
    assert result.passed is False
