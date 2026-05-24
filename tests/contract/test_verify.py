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
from rie_verify import FakeNliBackend, FakeSecondLlm, VerificationService


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
