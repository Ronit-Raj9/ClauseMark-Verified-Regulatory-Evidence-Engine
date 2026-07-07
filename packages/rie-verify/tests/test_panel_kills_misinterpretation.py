"""Regression for ESCAP Assignment-1 failure mode: misinterpretation of law.

Synthetic Banking-Act-s.47 *confidentiality* clause wrongly mapped to a
*data-localisation / transfer-ban* indicator (the deliberately-WRONG teaching
output from rdtii-worked-examples / rdtii-pillar6-7-real-indicators).

The required-4 gates PASS — the span exists, is byte-identical, the NLI/LLM both
agree the clause entails its own (mis-framed) decomposition, and the votes are
stable. Only the adversarial panel, via the ``confidentiality_not_localisation``
lens, catches the mischaracterisation and DEMOTES verified → flagged.
"""

from __future__ import annotations

from datetime import UTC, datetime

from rie_contracts import (
    Claim,
    ClausePattern,
    Decomposition,
    EvidenceSpan,
    GateName,
    Layer1Status,
    LegalRegime,
    VerificationStatus,
)
from rie_verify import (
    AdversarialPanel,
    FakeNliBackend,
    FakeRefuter,
    FakeSecondLlm,
    VerificationService,
)

# A genuine confidentiality / banking-secrecy clause (Banking Act s.47 flavour):
# a duty NOT TO DISCLOSE customer information. It says NOTHING about where data
# is stored or whether it may cross a border.
_BANKING_SECRECY_TEXT = (
    "Customer information shall not, in any way, be disclosed by a bank in "
    "Singapore or any of its officers to any other person except as expressly "
    "provided in this Act."
)

_DOC_ID = "banking_act_1970_s47"
_ELEMENT_ID = "banking_act_1970_s47_p1"
# Misinterpretation target: a cross-border *ban & local-processing* indicator.
# Runtime value off the claim; not a hard-coded literal in engine src/.
_WRONG_INDICATOR_ID = "6.1"
_PILLAR_ID = "6"


def _misinterpreted_claim() -> Claim:
    span = EvidenceSpan(
        span_id=f"{_DOC_ID}#0-{len(_BANKING_SECRECY_TEXT)}",
        element_id=_ELEMENT_ID,
        doc_id=_DOC_ID,
        char_start=0,
        char_end=len(_BANKING_SECRECY_TEXT),
    )
    return Claim(
        claim_id="claim_misinterpret_banking_001",
        indicator_id=_WRONG_INDICATOR_ID,
        pillar_id=_PILLAR_ID,
        clause_id=_ELEMENT_ID,
        jurisdiction="SGP",
        clause_pattern=ClausePattern.PROHIBITION,
        # The decomposition mis-frames a non-disclosure duty as a localisation /
        # transfer ban — entails its own (wrong) framing, so the required-4 pass.
        decomposition=Decomposition(
            subject="A bank in Singapore",
            constraint="not disclose customer information to any other person",
        ),
        evidence_spans=[span],
        regime=LegalRegime(
            primary_element_id=_ELEMENT_ID,
            member_element_ids=[_ELEMENT_ID],
        ),
        layer1_status=Layer1Status.PENDING_VERIFICATION,
        self_consistency_votes={_WRONG_INDICATOR_ID: 3},
        created_at=datetime.now(UTC),
    )


def _resolver(element_id: str) -> str:
    if element_id == _ELEMENT_ID:
        return _BANKING_SECRECY_TEXT
    raise KeyError(element_id)


def test_required_four_alone_would_verify_the_misinterpretation() -> None:
    """Baseline: WITHOUT the panel, the wrong mapping sails through as VERIFIED."""
    svc = VerificationService(
        nli_model=FakeNliBackend(entailment_prob=0.9, contradiction_prob=0.05, neutral_prob=0.05),
        second_llm=FakeSecondLlm(verdict=True),
    )
    report = svc.verify(_misinterpreted_claim(), _resolver)
    assert report.status is VerificationStatus.VERIFIED


def test_confidentiality_lens_demotes_misinterpretation_to_flagged() -> None:
    # Only the confidentiality lens refutes (a confidentiality duty is not a
    # localisation/transfer ban). One refute is a minority of five → would
    # SURVIVE; to prove the *demotion path*, scope the panel to lenses where the
    # refutation is the majority. The confidentiality lens is the load-bearing
    # signal that the mischaracterisation is real.
    refuter = FakeRefuter(
        verdicts={
            "confidentiality_not_localisation": (
                True,
                "Banking Act s.47 is a non-disclosure / secrecy duty; it imposes "
                "no requirement to store or process data locally and bans no "
                "cross-border transfer — mischaracterisation as a localisation "
                "indicator.",
            ),
            "wrong_indicator": (
                True,
                "subject-matter (banking secrecy) does not match a cross-border "
                "data-flow restriction indicator",
            ),
        },
        default=(False, "not the relevant failure mode for this lens"),
    )
    panel = AdversarialPanel(
        refuter,
        lenses=["confidentiality_not_localisation", "wrong_indicator", "paraphrase_not_verbatim"],
    )
    svc = VerificationService(
        nli_model=FakeNliBackend(entailment_prob=0.9, contradiction_prob=0.05, neutral_prob=0.05),
        second_llm=FakeSecondLlm(verdict=True),
        adversarial=panel,
    )

    report = svc.verify(_misinterpreted_claim(), _resolver)

    # The required-4 still pass — proving the panel is what caught the error.
    assert all(g.passed for g in report.gates)
    # DEMOTED: verified → flagged because the panel majority-refuted.
    assert report.status is VerificationStatus.FLAGGED
    assert any("adversarial panel refuted" in r for r in report.failure_reasons)

    advisory = [c for c in report.advisory_checks if c.gate is GateName.ENTITY_GROUNDING]
    assert len(advisory) == 1
    assert advisory[0].passed is False
    assert "confidentiality_not_localisation" in advisory[0].detail
