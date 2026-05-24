"""Full 4-gate `verify(...)` driver — status derivation matrix."""

from __future__ import annotations

from collections.abc import Callable

from rie_contracts import Claim, EvidenceSpan, GateName, VerificationStatus
from rie_verify import FakeNliBackend, FakeSecondLlm, VerificationService


def test_full_verify_all_pass_yields_verified(
    claim_factory: Callable[..., Claim],
    resolver: Callable[[str], str],
) -> None:
    svc = VerificationService(
        nli_model=FakeNliBackend(entailment_prob=0.9, contradiction_prob=0.05, neutral_prob=0.05),
        second_llm=FakeSecondLlm(verdict=True),
    )
    claim = claim_factory(self_consistency_votes={"6.4": 3})
    report = svc.verify(claim, resolver)
    assert report.status is VerificationStatus.VERIFIED
    assert len(report.gates) == 4
    assert all(g.passed for g in report.gates)
    assert report.failure_reasons == []
    assert {g.gate for g in report.gates} == {
        GateName.SPAN_EXISTENCE,
        GateName.VERBATIM_MATCH,
        GateName.ENTAILMENT,
        GateName.SELF_CONSISTENCY,
    }


def test_full_verify_entailment_disagreement_yields_flagged(
    claim_factory: Callable[..., Claim],
    resolver: Callable[[str], str],
) -> None:
    svc = VerificationService(
        nli_model=FakeNliBackend(entailment_prob=0.9, contradiction_prob=0.05, neutral_prob=0.05),
        second_llm=FakeSecondLlm(verdict=False),  # disagreement → flagged
    )
    claim = claim_factory(self_consistency_votes={"6.4": 3})
    report = svc.verify(claim, resolver)
    assert report.status is VerificationStatus.FLAGGED
    assert any("entailment" in r and "NLI/LLM disagree" in r for r in report.failure_reasons)


def test_full_verify_broken_offsets_yields_rejected(
    claim_factory: Callable[..., Claim],
    resolver: Callable[[str], str],
    element_text: str,
) -> None:
    svc = VerificationService(
        nli_model=FakeNliBackend(entailment_prob=0.9, contradiction_prob=0.05, neutral_prob=0.05),
        second_llm=FakeSecondLlm(verdict=True),
    )
    overshoot = len(element_text) + 100
    bad = EvidenceSpan(
        span_id=f"sample_dpa_2020#0-{overshoot}",
        element_id="sample_dpa_2020_s26_p1",
        doc_id="sample_dpa_2020",
        char_start=0,
        char_end=overshoot,
    )
    claim = claim_factory(evidence_spans=[bad], self_consistency_votes={"6.4": 3})
    report = svc.verify(claim, resolver)
    assert report.status is VerificationStatus.REJECTED
    # The deterministic failure must be in the failure_reasons.
    assert any("span_existence" in r for r in report.failure_reasons)


def test_full_verify_unstable_votes_yield_flagged(
    claim_factory: Callable[..., Claim],
    resolver: Callable[[str], str],
) -> None:
    svc = VerificationService(
        nli_model=FakeNliBackend(entailment_prob=0.9, contradiction_prob=0.05, neutral_prob=0.05),
        second_llm=FakeSecondLlm(verdict=True),
    )
    claim = claim_factory(self_consistency_votes={"6.4": 1, "6.3": 1, "7.2": 1})
    report = svc.verify(claim, resolver)
    assert report.status is VerificationStatus.FLAGGED
    assert any("self_consistency" in r for r in report.failure_reasons)


def test_full_verify_deterministic_failure_dominates_model_failure(
    claim_factory: Callable[..., Claim],
    resolver: Callable[[str], str],
    element_text: str,
) -> None:
    """A claim with both deterministic and model failures must be REJECTED
    (not FLAGGED) — deterministic failure is the hardest signal."""
    svc = VerificationService(
        nli_model=FakeNliBackend(entailment_prob=0.05, contradiction_prob=0.9, neutral_prob=0.05),
        second_llm=FakeSecondLlm(verdict=False),
    )
    overshoot = len(element_text) + 100
    bad = EvidenceSpan(
        span_id=f"sample_dpa_2020#0-{overshoot}",
        element_id="sample_dpa_2020_s26_p1",
        doc_id="sample_dpa_2020",
        char_start=0,
        char_end=overshoot,
    )
    claim = claim_factory(
        evidence_spans=[bad],
        self_consistency_votes={"6.4": 1, "6.3": 1, "7.2": 1},
    )
    report = svc.verify(claim, resolver)
    assert report.status is VerificationStatus.REJECTED
