"""Adversarial refutation panel — DEMOTE-only contract.

Proves: all-survive → score 1.0, verdict survive; majority-refute demotes a
VERIFIED claim to FLAGGED; advisory_checks populated; NEVER promotes a REJECTED.
"""

from __future__ import annotations

from collections.abc import Callable

from rie_contracts import Claim, EvidenceSpan, GateName, VerificationStatus
from rie_verify import (
    DEFAULT_LENSES,
    AdversarialPanel,
    FakeNliBackend,
    FakeRefuter,
    FakeSecondLlm,
    VerificationService,
)


def _all_survive_refuter() -> FakeRefuter:
    return FakeRefuter(default=(False, "no issue under this lens"))


def _all_refute_refuter() -> FakeRefuter:
    return FakeRefuter(default=(True, "this mapping is wrong under this lens"))


# ── Panel unit behaviour ────────────────────────────────────────────────────


def test_panel_all_survive_scores_one_and_passes() -> None:
    panel = AdversarialPanel(_all_survive_refuter())
    result = panel.run(
        claim_text="An organisation must do X.",
        indicator_def="some indicator definition",
        span_text="the cited legal text",
    )
    assert result.gate is GateName.ENTITY_GROUNDING
    assert result.passed is True
    assert result.score == 1.0


def test_panel_majority_refute_fails_and_lists_lenses() -> None:
    panel = AdversarialPanel(_all_refute_refuter())
    result = panel.run(
        claim_text="An organisation must do X.",
        indicator_def="some indicator definition",
        span_text="the cited legal text",
    )
    assert result.passed is False
    assert result.score == 0.0
    # detail names the refuting lenses
    for lens in DEFAULT_LENSES:
        assert lens in result.detail


def test_panel_minority_refute_still_survives() -> None:
    # Only ONE lens refutes out of five → minority → survives.
    refuter = FakeRefuter(
        verdicts={"wrong_indicator": (True, "looks like a different indicator")},
        default=(False, "fine"),
    )
    panel = AdversarialPanel(refuter)
    outcome = panel.evaluate("claim", "def", "span")
    assert outcome.survives is True
    assert outcome.refuting_lenses == ("wrong_indicator",)
    assert outcome.score == 4 / 5


def test_panel_exact_majority_refutes() -> None:
    # 3 of 5 refute → majority → does NOT survive.
    refuter = FakeRefuter(
        verdicts={
            "wrong_indicator": (True, "a"),
            "confidentiality_not_localisation": (True, "b"),
            "government_data_scope_exception": (True, "c"),
        },
        default=(False, "fine"),
    )
    panel = AdversarialPanel(refuter)
    outcome = panel.evaluate("claim", "def", "span")
    assert outcome.survives is False
    assert outcome.score == 2 / 5


# ── Service integration: DEMOTE-only ────────────────────────────────────────


def test_panel_survive_leaves_verified_intact(
    claim_factory: Callable[..., Claim],
    resolver: Callable[[str], str],
) -> None:
    svc = VerificationService(
        nli_model=FakeNliBackend(entailment_prob=0.9, contradiction_prob=0.05, neutral_prob=0.05),
        second_llm=FakeSecondLlm(verdict=True),
        adversarial=AdversarialPanel(_all_survive_refuter()),
    )
    report = svc.verify(claim_factory(), resolver)
    assert report.status is VerificationStatus.VERIFIED
    # Advisory channel populated with the panel result.
    assert len(report.advisory_checks) == 1
    assert report.advisory_checks[0].gate is GateName.ENTITY_GROUNDING
    assert report.advisory_checks[0].passed is True


def test_panel_majority_refute_demotes_verified_to_flagged(
    claim_factory: Callable[..., Claim],
    resolver: Callable[[str], str],
) -> None:
    svc = VerificationService(
        nli_model=FakeNliBackend(entailment_prob=0.9, contradiction_prob=0.05, neutral_prob=0.05),
        second_llm=FakeSecondLlm(verdict=True),
        adversarial=AdversarialPanel(_all_refute_refuter()),
    )
    report = svc.verify(claim_factory(), resolver)
    # The required-4 all passed → would be VERIFIED, but the panel DEMOTES it.
    assert report.status is VerificationStatus.FLAGGED
    assert any("adversarial panel refuted" in r for r in report.failure_reasons)
    # The required-4 gates themselves still all pass — only status was demoted.
    assert all(g.passed for g in report.gates)
    assert len(report.advisory_checks) == 1
    assert report.advisory_checks[0].passed is False


def test_panel_never_promotes_a_rejected_claim(
    claim_factory: Callable[..., Claim],
    resolver: Callable[[str], str],
    element_text: str,
) -> None:
    # Deterministic failure → REJECTED. An all-SURVIVE panel must NOT promote it.
    svc = VerificationService(
        nli_model=FakeNliBackend(entailment_prob=0.9, contradiction_prob=0.05, neutral_prob=0.05),
        second_llm=FakeSecondLlm(verdict=True),
        adversarial=AdversarialPanel(_all_survive_refuter()),
    )
    overshoot = len(element_text) + 100
    bad = EvidenceSpan(
        span_id=f"sample_dpa_2020#0-{overshoot}",
        element_id="sample_dpa_2020_s26_p1",
        doc_id="sample_dpa_2020",
        char_start=0,
        char_end=overshoot,
    )
    report = svc.verify(claim_factory(evidence_spans=[bad]), resolver)
    assert report.status is VerificationStatus.REJECTED


def test_panel_does_not_re_demote_an_already_flagged_claim(
    claim_factory: Callable[..., Claim],
    resolver: Callable[[str], str],
) -> None:
    # Model-gate failure → FLAGGED. An all-REFUTE panel keeps it FLAGGED
    # (cannot promote, and there is nothing to demote past FLAGGED).
    svc = VerificationService(
        nli_model=FakeNliBackend(entailment_prob=0.9, contradiction_prob=0.05, neutral_prob=0.05),
        second_llm=FakeSecondLlm(verdict=False),  # disagreement → FLAGGED
        adversarial=AdversarialPanel(_all_refute_refuter()),
    )
    report = svc.verify(claim_factory(), resolver)
    assert report.status is VerificationStatus.FLAGGED
    # Panel result still recorded in advisory_checks.
    assert any(c.gate is GateName.ENTITY_GROUNDING for c in report.advisory_checks)


def test_panel_default_to_refute_when_skeptic_raises() -> None:
    class _RaisingRefuter:
        def refute(
            self, claim_text: str, indicator_def: str, span_text: str, lens: str
        ) -> tuple[bool, str]:
            raise RuntimeError("model down")

    panel = AdversarialPanel(_RaisingRefuter())
    outcome = panel.evaluate("claim", "def", "span")
    # Every lens defaulted to refute → majority → does NOT survive.
    assert outcome.survives is False
    assert all(v.refuted for v in outcome.verdicts)
