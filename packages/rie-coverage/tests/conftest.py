"""Local fixtures for the rie-coverage test suite.

Builds tiny but contract-valid ``Claim`` instances so each test can focus on
the coverage policy, not on assembling Pydantic models.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from rie_contracts import (
    Claim,
    ClausePattern,
    Decomposition,
    EvidenceSpan,
    Layer1Status,
    LegalRegime,
)


@pytest.fixture
def now() -> datetime:
    return datetime(2026, 5, 24, tzinfo=UTC)


def make_claim(
    *,
    claim_id: str = "claim_001",
    indicator_id: str = "6.4",
    pillar_id: str = "6",
    jurisdiction: str = "SAMPLE",
    layer1_status: Layer1Status = Layer1Status.VERIFIED,
    clause_id: str = "sample_dpa_2020_s26_p1",
    doc_id: str = "sample_dpa_2020",
    char_start: int = 4120,
    char_end: int = 4215,
    created_at: datetime | None = None,
) -> Claim:
    """Build a contract-valid Claim. All knobs default to a VERIFIED 6.4 claim."""
    span = EvidenceSpan(
        span_id=f"{doc_id}#{char_start}-{char_end}",
        element_id=clause_id,
        doc_id=doc_id,
        char_start=char_start,
        char_end=char_end,
    )
    return Claim(
        claim_id=claim_id,
        indicator_id=indicator_id,
        pillar_id=pillar_id,
        clause_id=clause_id,
        jurisdiction=jurisdiction,
        clause_pattern=ClausePattern.CONDITIONAL_REGIME,
        decomposition=Decomposition(
            subject="organisation",
            condition="transfer outside the country",
            constraint="recipient bound by comparable-protection obligations",
        ),
        evidence_spans=[span],
        regime=LegalRegime(
            primary_element_id=clause_id,
            member_element_ids=[clause_id],
        ),
        layer1_status=layer1_status,
        created_at=created_at or datetime(2026, 5, 24, tzinfo=UTC),
    )


@pytest.fixture
def make_claim_fixture():
    """Expose ``make_claim`` as a fixture for parametrised tests."""
    return make_claim
