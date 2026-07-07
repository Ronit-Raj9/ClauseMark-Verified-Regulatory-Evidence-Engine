"""Source re-fetch — ADVISORY URL liveness; flags but never blocks alone."""

from __future__ import annotations

from collections.abc import Callable

from rie_contracts import Claim, GateName, VerificationStatus
from rie_verify import (
    FakeNliBackend,
    FakeSecondLlm,
    FakeSourceFetcher,
    VerificationService,
    verify_source_live,
)

LIVE_URL = "https://laws.example.gov/dpa/2020/s26"
DEAD_URL = "https://laws.example.gov/dpa/2020/moved"


# ── Unit ────────────────────────────────────────────────────────────────────


def test_verify_source_live_hit_passes() -> None:
    fetcher = FakeSourceFetcher({LIVE_URL: "An organisation shall not transfer..."})
    result = verify_source_live(LIVE_URL, fetcher)
    assert result.gate is GateName.ENTITY_GROUNDING
    assert result.passed is True
    assert result.score == 1.0


def test_verify_source_live_miss_flags() -> None:
    fetcher = FakeSourceFetcher({DEAD_URL: None})
    result = verify_source_live(DEAD_URL, fetcher)
    assert result.passed is False
    assert "dead/changed URL" in result.detail


def test_verify_source_live_unknown_url_flags() -> None:
    fetcher = FakeSourceFetcher({})  # url absent → None → dead
    result = verify_source_live(DEAD_URL, fetcher)
    assert result.passed is False


def test_verify_source_live_none_url_flags() -> None:
    fetcher = FakeSourceFetcher({})
    result = verify_source_live(None, fetcher)
    assert result.passed is False
    assert "no source_url" in result.detail


# ── Service integration: advisory, status unchanged by it alone ─────────────


def test_service_source_hit_keeps_verified(
    claim_factory: Callable[..., Claim],
    resolver: Callable[[str], str],
) -> None:
    svc = VerificationService(
        nli_model=FakeNliBackend(entailment_prob=0.9, contradiction_prob=0.05, neutral_prob=0.05),
        second_llm=FakeSecondLlm(verdict=True),
        source_fetcher=FakeSourceFetcher({LIVE_URL: "real law text"}),
    )
    report = svc.verify(claim_factory(), resolver, source_url=LIVE_URL)
    assert report.status is VerificationStatus.VERIFIED
    advisory = [c for c in report.advisory_checks if c.gate is GateName.ENTITY_GROUNDING]
    assert len(advisory) == 1
    assert advisory[0].passed is True


def test_service_dead_source_does_not_block_alone(
    claim_factory: Callable[..., Claim],
    resolver: Callable[[str], str],
) -> None:
    # Required-4 all pass; only the source URL is dead. Status stays VERIFIED;
    # the advisory carries the (non-blocking) flag.
    svc = VerificationService(
        nli_model=FakeNliBackend(entailment_prob=0.9, contradiction_prob=0.05, neutral_prob=0.05),
        second_llm=FakeSecondLlm(verdict=True),
        source_fetcher=FakeSourceFetcher({DEAD_URL: None}),
    )
    report = svc.verify(claim_factory(), resolver, source_url=DEAD_URL)
    assert report.status is VerificationStatus.VERIFIED
    advisory = [c for c in report.advisory_checks if c.gate is GateName.ENTITY_GROUNDING]
    assert len(advisory) == 1
    assert advisory[0].passed is False
    # The dead URL did NOT add a required-4 failure reason.
    assert report.failure_reasons == []
