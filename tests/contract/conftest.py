"""Shared fixtures + Pydantic factories for contract tests across adapters."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from rie_contracts import (
    AuthorityTier,
    BoundingBox,
    Claim,
    ClausePattern,
    Decomposition,
    DocumentMeta,
    DocumentType,
    Element,
    ElementType,
    EvidenceSpan,
    Layer1Status,
    LegalRegime,
)


@pytest.fixture
def now() -> datetime:
    return datetime(2026, 5, 24, tzinfo=UTC)


@pytest.fixture
def doc_meta(now: datetime) -> DocumentMeta:
    return DocumentMeta(
        doc_id="sample_dpa_2020",
        jurisdiction="SAMPLE",
        title="Sample Personal Data Protection Act, 2020",
        document_type=DocumentType.STATUTE,
        authority_tier=AuthorityTier.TIER_1_STATUTE,
        sha256="a" * 64,
        retrieved_at=now,
        language="en",
    )


@pytest.fixture
def element() -> Element:
    text = (
        "An organisation shall not transfer any personal data outside the country "
        "unless it has taken appropriate steps to ensure that the recipient is "
        "bound by legally enforceable obligations to provide to the transferred "
        "personal data a standard of protection that is comparable to the "
        "protection under this Act."
    )
    return Element(
        element_id="sample_dpa_2020_s26_p1",
        doc_id="sample_dpa_2020",
        parent_id="sample_dpa_2020_s26",
        element_type=ElementType.PARAGRAPH,
        text=text,
        page=4,
        bbox=BoundingBox(page=4, x0=72.0, y0=120.0, x1=540.0, y1=240.0),
        char_start=4120,
        char_end=4120 + len(text),
        extraction_confidence=0.99,
    )


@pytest.fixture
def claim(now: datetime, element: Element) -> Claim:
    return Claim(
        claim_id="claim_sample_001",
        indicator_id="6.4",
        pillar_id="6",
        clause_id=element.element_id,
        jurisdiction="SAMPLE",
        clause_pattern=ClausePattern.CONDITIONAL_REGIME,
        decomposition=Decomposition(
            subject="organisation",
            condition="transfer outside the country",
            constraint="recipient bound by comparable-protection obligations",
        ),
        evidence_spans=[
            EvidenceSpan(
                span_id=f"{element.doc_id}#{element.char_start}-{element.char_end}",
                element_id=element.element_id,
                doc_id=element.doc_id,
                char_start=element.char_start,
                char_end=element.char_end,
            )
        ],
        regime=LegalRegime(
            primary_element_id=element.element_id,
            member_element_ids=[element.element_id, "sample_dpa_2020_s2_def_personal_data"],
        ),
        layer1_status=Layer1Status.PENDING_VERIFICATION,
        created_at=now,
    )
