"""Shared fixtures for the `rie-verify` test suite.

Spans here are stored *element-relative* so direct slicing
`element_text[span.char_start : span.char_end]` reproduces the cited
substring — matches the offset convention documented in
`rie_verify.service`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
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

ELEMENT_ID = "sample_dpa_2020_s26_p1"
DOC_ID = "sample_dpa_2020"
INDICATOR_ID = "6.4"
PILLAR_ID = "6"

ELEMENT_TEXT = (
    "An organisation shall not transfer any personal data outside the country "
    "unless it has taken appropriate steps to ensure that the recipient is "
    "bound by legally enforceable obligations to provide to the transferred "
    "personal data a standard of protection that is comparable to the "
    "protection under this Act."
)


@pytest.fixture
def element_text() -> str:
    return ELEMENT_TEXT


@pytest.fixture
def text_store() -> dict[str, str]:
    return {ELEMENT_ID: ELEMENT_TEXT}


@pytest.fixture
def resolver(text_store: Mapping[str, str]) -> Callable[[str], str]:
    def _resolver(element_id: str) -> str:
        if element_id not in text_store:
            raise KeyError(element_id)
        return text_store[element_id]

    return _resolver


@pytest.fixture
def now() -> datetime:
    return datetime(2026, 5, 24, tzinfo=UTC)


@pytest.fixture
def good_span() -> EvidenceSpan:
    """A span covering the whole element text — element-relative offsets."""
    char_start = 0
    char_end = len(ELEMENT_TEXT)
    return EvidenceSpan(
        span_id=f"{DOC_ID}#{char_start}-{char_end}",
        element_id=ELEMENT_ID,
        doc_id=DOC_ID,
        char_start=char_start,
        char_end=char_end,
    )


@pytest.fixture
def claim_factory(
    now: datetime,
    good_span: EvidenceSpan,
) -> Callable[..., Claim]:
    def _build(
        *,
        indicator_id: str = INDICATOR_ID,
        evidence_spans: list[EvidenceSpan] | None = None,
        self_consistency_votes: dict[str, int] | None = None,
    ) -> Claim:
        spans = evidence_spans if evidence_spans is not None else [good_span]
        votes = self_consistency_votes if self_consistency_votes is not None else {INDICATOR_ID: 3}
        return Claim(
            claim_id="claim_sample_001",
            indicator_id=indicator_id,
            pillar_id=PILLAR_ID,
            clause_id=ELEMENT_ID,
            jurisdiction="SAMPLE",
            clause_pattern=ClausePattern.CONDITIONAL_REGIME,
            decomposition=Decomposition(
                subject="An organisation",
                condition="transferring personal data outside the country",
                constraint=(
                    "ensure the recipient is bound by legally enforceable "
                    "obligations providing comparable protection"
                ),
            ),
            evidence_spans=spans,
            regime=LegalRegime(
                primary_element_id=ELEMENT_ID,
                member_element_ids=[ELEMENT_ID],
            ),
            layer1_status=Layer1Status.PENDING_VERIFICATION,
            self_consistency_votes=votes,
            created_at=now,
        )

    return _build
