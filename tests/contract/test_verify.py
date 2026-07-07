"""Contract test for `VerifierPort` — any conforming implementation passes.

`rie_verify.VerificationService` is the in-tree implementation under test.
The test also exercises `isinstance(..., VerifierPort)` at runtime because
`VerifierPort` is `@runtime_checkable`.
"""

from __future__ import annotations

from datetime import UTC, datetime

from rie_contracts import (
    Claim,
    ClausePattern,
    Decomposition,
    Element,
    EvidenceSpan,
    GateName,
    Layer1Status,
    LegalRegime,
    VerificationStatus,
    VerifierPort,
)
from rie_verify import (
    AdversarialPanel,
    EntityGroundingChecker,
    FakeEntityExtractor,
    FakeNliBackend,
    FakeRefuter,
    FakeSecondLlm,
    FakeSourceFetcher,
    GroundingEntity,
    VerificationService,
)


def _build_claim(element: Element) -> Claim:
    # Spans are stored *element-relative* per `rie_verify.service` convention,
    # which is what the contract test exercises.
    span = EvidenceSpan(
        span_id=f"{element.doc_id}#0-{len(element.text)}",
        element_id=element.element_id,
        doc_id=element.doc_id,
        char_start=0,
        char_end=len(element.text),
    )
    return Claim(
        claim_id="claim_contract_001",
        indicator_id="6.4",
        pillar_id="6",
        clause_id=element.element_id,
        jurisdiction="SAMPLE",
        clause_pattern=ClausePattern.CONDITIONAL_REGIME,
        decomposition=Decomposition(
            subject="organisation",
            condition="transferring personal data outside the country",
            constraint="ensure recipient is bound by comparable-protection obligations",
        ),
        evidence_spans=[span],
        regime=LegalRegime(
            primary_element_id=element.element_id,
            member_element_ids=[element.element_id],
        ),
        layer1_status=Layer1Status.PENDING_VERIFICATION,
        self_consistency_votes={"6.4": 3},
        created_at=datetime.now(UTC),
    )


def test_verification_service_is_runtime_verifier_port() -> None:
    svc = VerificationService(
        nli_model=FakeNliBackend(entailment_prob=0.9),
        second_llm=FakeSecondLlm(verdict=True),
    )
    assert isinstance(svc, VerifierPort)


def test_verification_service_with_advisories_is_still_verifier_port() -> None:
    # Wiring the ADVISORY adversarial panel + source fetcher MUST NOT break the
    # VerifierPort surface — verify(claim, resolver) stays a valid call.
    svc = VerificationService(
        nli_model=FakeNliBackend(entailment_prob=0.9),
        second_llm=FakeSecondLlm(verdict=True),
        adversarial=AdversarialPanel(FakeRefuter(default=(False, "ok"))),
        source_fetcher=FakeSourceFetcher({}),
    )
    assert isinstance(svc, VerifierPort)


def test_verification_service_full_pass(element: Element) -> None:
    svc = VerificationService(
        nli_model=FakeNliBackend(entailment_prob=0.9, contradiction_prob=0.05, neutral_prob=0.05),
        second_llm=FakeSecondLlm(verdict=True),
    )
    claim = _build_claim(element)
    text_store = {element.element_id: element.text}

    def resolver(element_id: str) -> str:
        if element_id not in text_store:
            raise KeyError(element_id)
        return text_store[element_id]

    report = svc.verify(claim, resolver)
    assert report.status is VerificationStatus.VERIFIED
    assert len(report.gates) == 4
    assert {g.gate for g in report.gates} == {
        GateName.SPAN_EXISTENCE,
        GateName.VERBATIM_MATCH,
        GateName.ENTAILMENT,
        GateName.SELF_CONSISTENCY,
    }
    assert all(g.passed for g in report.gates)
    assert report.failure_reasons == []


def test_verification_service_run_gate_returns_single_gate_result(
    element: Element,
) -> None:
    svc = VerificationService(
        nli_model=FakeNliBackend(entailment_prob=0.9),
        second_llm=FakeSecondLlm(verdict=True),
    )
    claim = _build_claim(element)
    text_store = {element.element_id: element.text}

    def resolver(element_id: str) -> str:
        return text_store[element_id]

    for gate_name in (
        GateName.SPAN_EXISTENCE,
        GateName.VERBATIM_MATCH,
        GateName.ENTAILMENT,
        GateName.SELF_CONSISTENCY,
    ):
        result = svc.run_gate(gate_name, claim, resolver)
        assert result.gate is gate_name
        assert result.passed is True


def test_advisory_checks_empty_when_no_grounder(element: Element) -> None:
    svc = VerificationService(
        nli_model=FakeNliBackend(entailment_prob=0.9, contradiction_prob=0.05, neutral_prob=0.05),
        second_llm=FakeSecondLlm(verdict=True),
    )
    claim = _build_claim(element)
    text_store = {element.element_id: element.text}

    report = svc.verify(claim, lambda eid: text_store[eid])
    assert report.advisory_checks == []


def test_advisory_checks_populated_when_grounder_set(element: Element) -> None:
    grounder = EntityGroundingChecker(
        FakeEntityExtractor(
            [GroundingEntity(text="organisation", label="ORG", char_start=0, char_end=12)]
        )
    )
    svc = VerificationService(
        nli_model=FakeNliBackend(entailment_prob=0.9, contradiction_prob=0.05, neutral_prob=0.05),
        second_llm=FakeSecondLlm(verdict=True),
        entity_grounder=grounder,
    )
    claim = _build_claim(element)
    text_store = {element.element_id: element.text}

    report = svc.verify(claim, lambda eid: text_store[eid])
    # Advisory populated, but status remains determined solely by required-4.
    assert report.status is VerificationStatus.VERIFIED
    assert len(report.advisory_checks) == 1
    assert report.advisory_checks[0].gate is GateName.ENTITY_GROUNDING
    assert report.advisory_checks[0].gate not in {g.gate for g in report.gates}


def test_adversarial_panel_can_demote_but_never_promote(element: Element) -> None:
    text_store = {element.element_id: element.text}

    def resolver(element_id: str) -> str:
        return text_store[element_id]

    # All-refute panel DEMOTES a would-be VERIFIED claim to FLAGGED.
    demoting = VerificationService(
        nli_model=FakeNliBackend(entailment_prob=0.9, contradiction_prob=0.05, neutral_prob=0.05),
        second_llm=FakeSecondLlm(verdict=True),
        adversarial=AdversarialPanel(FakeRefuter(default=(True, "refuted"))),
    )
    report = demoting.verify(_build_claim(element), resolver)
    assert report.status is VerificationStatus.FLAGGED
    assert all(g.passed for g in report.gates)  # required-4 untouched
    assert report.advisory_checks[0].gate is GateName.ENTITY_GROUNDING

    # All-survive panel leaves a VERIFIED claim VERIFIED (never promotes either).
    surviving = VerificationService(
        nli_model=FakeNliBackend(entailment_prob=0.9, contradiction_prob=0.05, neutral_prob=0.05),
        second_llm=FakeSecondLlm(verdict=True),
        adversarial=AdversarialPanel(FakeRefuter(default=(False, "ok"))),
    )
    report2 = surviving.verify(_build_claim(element), resolver)
    assert report2.status is VerificationStatus.VERIFIED


def test_source_refetch_advisory_does_not_block(element: Element) -> None:
    text_store = {element.element_id: element.text}

    def resolver(element_id: str) -> str:
        return text_store[element_id]

    svc = VerificationService(
        nli_model=FakeNliBackend(entailment_prob=0.9, contradiction_prob=0.05, neutral_prob=0.05),
        second_llm=FakeSecondLlm(verdict=True),
        source_fetcher=FakeSourceFetcher({"https://dead.example": None}),
    )
    report = svc.verify(_build_claim(element), resolver, source_url="https://dead.example")
    # Dead URL is advisory-only — required-4 pass → stays VERIFIED.
    assert report.status is VerificationStatus.VERIFIED
    assert any(
        c.gate is GateName.ENTITY_GROUNDING and c.passed is False for c in report.advisory_checks
    )
