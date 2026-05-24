"""Gate 4 — self_consistency. N=3 votes on FIXED input; unstable → fail."""

from __future__ import annotations

from collections.abc import Callable

from rie_contracts import Claim, GateName
from rie_verify import FakeNliBackend, FakeSecondLlm, VerificationService


def _service() -> VerificationService:
    return VerificationService(
        nli_model=FakeNliBackend(),
        second_llm=FakeSecondLlm(verdict=True),
    )


def test_self_consistency_passes_on_two_thirds_majority_for_claim_indicator(
    claim_factory: Callable[..., Claim],
) -> None:
    claim = claim_factory(
        indicator_id="6.4",
        self_consistency_votes={"6.4": 2, "6.3": 1},
    )
    result = _service().run_gate(
        GateName.SELF_CONSISTENCY,
        claim,
        lambda _id: "",  # resolver unused
    )
    assert result.passed is True
    assert result.score is not None and 0.66 < result.score < 0.67


def test_self_consistency_fails_on_three_way_split(
    claim_factory: Callable[..., Claim],
) -> None:
    claim = claim_factory(
        indicator_id="6.4",
        self_consistency_votes={"6.4": 1, "6.3": 1, "7.2": 1},
    )
    result = _service().run_gate(GateName.SELF_CONSISTENCY, claim, lambda _id: "")
    assert result.passed is False
    assert "unstable" in result.detail


def test_self_consistency_fails_when_winner_disagrees_with_claim_indicator(
    claim_factory: Callable[..., Claim],
) -> None:
    claim = claim_factory(
        indicator_id="6.4",
        self_consistency_votes={"6.3": 2, "6.4": 1},
    )
    result = _service().run_gate(GateName.SELF_CONSISTENCY, claim, lambda _id: "")
    assert result.passed is False
    assert "unstable" in result.detail
    assert "claim.indicator_id" in result.detail


def test_self_consistency_fails_when_no_votes_recorded(
    claim_factory: Callable[..., Claim],
) -> None:
    claim = claim_factory(self_consistency_votes={})
    result = _service().run_gate(GateName.SELF_CONSISTENCY, claim, lambda _id: "")
    assert result.passed is False


def test_self_consistency_unanimous_pass(
    claim_factory: Callable[..., Claim],
) -> None:
    claim = claim_factory(
        indicator_id="6.4",
        self_consistency_votes={"6.4": 3},
    )
    result = _service().run_gate(GateName.SELF_CONSISTENCY, claim, lambda _id: "")
    assert result.passed is True
