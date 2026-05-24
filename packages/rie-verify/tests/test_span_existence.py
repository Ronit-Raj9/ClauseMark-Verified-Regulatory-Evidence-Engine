"""Gate 1 — span_existence. Deterministic; arithmetic over offsets."""

from __future__ import annotations

from collections.abc import Callable

from rie_contracts import Claim, EvidenceSpan, GateName
from rie_verify import FakeNliBackend, FakeSecondLlm, VerificationService


def _service() -> VerificationService:
    return VerificationService(
        nli_model=FakeNliBackend(),
        second_llm=FakeSecondLlm(verdict=True),
    )


def test_span_existence_passes_when_slice_is_valid(
    claim_factory: Callable[..., Claim],
    resolver: Callable[[str], str],
) -> None:
    claim = claim_factory()
    result = _service().run_gate(GateName.SPAN_EXISTENCE, claim, resolver)
    assert result.gate is GateName.SPAN_EXISTENCE
    assert result.passed is True


def test_span_existence_fails_when_char_end_exceeds_text_length(
    claim_factory: Callable[..., Claim],
    resolver: Callable[[str], str],
    element_text: str,
) -> None:
    overshoot = len(element_text) + 50
    bad_span = EvidenceSpan(
        span_id=f"sample_dpa_2020#0-{overshoot}",
        element_id="sample_dpa_2020_s26_p1",
        doc_id="sample_dpa_2020",
        char_start=0,
        char_end=overshoot,
    )
    claim = claim_factory(evidence_spans=[bad_span])
    result = _service().run_gate(GateName.SPAN_EXISTENCE, claim, resolver)
    assert result.passed is False
    assert "exceeds element" in result.detail


def test_span_existence_fails_when_char_start_greater_than_char_end() -> None:
    """EvidenceSpan model itself enforces char_end >= char_start.

    Build a *valid* span that places the slice at a zero-width tail position
    (char_start == char_end), which collapses to an empty slice and must fail
    the gate. This proves the slice-emptiness branch in the service.
    """
    # Construct a span via the model that satisfies its own invariants but
    # still produces an empty slice — the gate must reject it.
    empty_slice_span = EvidenceSpan(
        span_id="sample_dpa_2020#10-10",
        element_id="sample_dpa_2020_s26_p1",
        doc_id="sample_dpa_2020",
        char_start=10,
        char_end=10,
    )
    from rie_verify.service import VerificationService as VS

    from .conftest import ELEMENT_TEXT

    class _MapResolver:
        def __call__(self, _id: str) -> str:
            return ELEMENT_TEXT

    svc = VS(nli_model=FakeNliBackend(), second_llm=FakeSecondLlm(verdict=True))

    # We need a Claim shell — minimal valid one.
    from datetime import UTC, datetime

    from rie_contracts import (
        Claim,
        ClausePattern,
        Decomposition,
        Layer1Status,
        LegalRegime,
    )

    claim = Claim(
        claim_id="c",
        indicator_id="6.4",
        pillar_id="6",
        clause_id="sample_dpa_2020_s26_p1",
        jurisdiction="SAMPLE",
        clause_pattern=ClausePattern.CONDITIONAL_REGIME,
        decomposition=Decomposition(subject="s", constraint="c"),
        evidence_spans=[empty_slice_span],
        regime=LegalRegime(
            primary_element_id="sample_dpa_2020_s26_p1",
            member_element_ids=["sample_dpa_2020_s26_p1"],
        ),
        layer1_status=Layer1Status.PENDING_VERIFICATION,
        created_at=datetime.now(UTC),
    )

    result = svc.run_gate(GateName.SPAN_EXISTENCE, claim, _MapResolver())
    assert result.passed is False
    assert "empty" in result.detail


def test_span_existence_fails_when_element_unresolvable(
    claim_factory: Callable[..., Claim],
) -> None:
    def _empty_resolver(_id: str) -> str:
        raise KeyError(_id)

    claim = claim_factory()
    result = _service().run_gate(GateName.SPAN_EXISTENCE, claim, _empty_resolver)
    assert result.passed is False
    assert "not resolvable" in result.detail


def test_span_existence_fails_when_claim_has_no_spans(
    claim_factory: Callable[..., Claim],
    resolver: Callable[[str], str],
) -> None:
    # The Claim model permits an empty evidence_spans list; the verifier rejects it.
    claim = claim_factory(evidence_spans=[])
    result = _service().run_gate(GateName.SPAN_EXISTENCE, claim, resolver)
    assert result.passed is False
