"""Gate 2 — verbatim_match. Re-extraction must be byte-identical to itself."""

from __future__ import annotations

from collections.abc import Callable

from rie_contracts import Claim, GateName
from rie_verify import FakeNliBackend, FakeSecondLlm, VerificationService


def _service() -> VerificationService:
    return VerificationService(
        nli_model=FakeNliBackend(),
        second_llm=FakeSecondLlm(verdict=True),
    )


def test_verbatim_match_passes_when_resolver_is_stable(
    claim_factory: Callable[..., Claim],
    resolver: Callable[[str], str],
) -> None:
    claim = claim_factory()
    result = _service().run_gate(GateName.VERBATIM_MATCH, claim, resolver)
    assert result.passed is True


def test_verbatim_match_fails_when_resolver_flips_a_single_char(
    claim_factory: Callable[..., Claim],
    element_text: str,
) -> None:
    """Simulate data-integrity drift: the resolver returns a different string
    on the second call. The gate must catch the one-byte divergence."""
    counter = {"n": 0}
    corrupted = element_text[:50] + "X" + element_text[51:]  # flip one char

    def _flipping_resolver(_id: str) -> str:
        counter["n"] += 1
        return element_text if counter["n"] == 1 else corrupted

    claim = claim_factory()
    result = _service().run_gate(GateName.VERBATIM_MATCH, claim, _flipping_resolver)
    assert result.passed is False
    assert "byte-identical" in result.detail


def test_verbatim_match_fails_on_out_of_range_offsets(
    claim_factory: Callable[..., Claim],
    resolver: Callable[[str], str],
    element_text: str,
) -> None:
    from rie_contracts import EvidenceSpan

    overshoot = len(element_text) + 100
    bad = EvidenceSpan(
        span_id=f"sample_dpa_2020#0-{overshoot}",
        element_id="sample_dpa_2020_s26_p1",
        doc_id="sample_dpa_2020",
        char_start=0,
        char_end=overshoot,
    )
    claim = claim_factory(evidence_spans=[bad])
    result = _service().run_gate(GateName.VERBATIM_MATCH, claim, resolver)
    assert result.passed is False
