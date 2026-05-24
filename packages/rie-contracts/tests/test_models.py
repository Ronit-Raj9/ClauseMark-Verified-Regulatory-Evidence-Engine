"""Smoke tests for rie-contracts data models — they MUST stay frozen."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError
from rie_contracts import (
    AuthorityTier,
    BoundingBox,
    Claim,
    ClausePattern,
    CoverageRecord,
    CoverageState,
    Decomposition,
    DocumentMeta,
    DocumentType,
    Element,
    ElementType,
    EvidenceSpan,
    GateName,
    GateResult,
    Layer1Status,
    LegalRegime,
    SpanRole,
    VerificationReport,
    VerificationStatus,
)


def _now() -> datetime:
    return datetime(2026, 5, 24, tzinfo=UTC)


def test_evidence_span_id_must_match_offsets() -> None:
    EvidenceSpan(
        span_id="doc1#10-20",
        element_id="e1",
        doc_id="doc1",
        char_start=10,
        char_end=20,
        role=SpanRole.PRIMARY,
    )
    with pytest.raises(ValidationError):
        EvidenceSpan(
            span_id="doc1#10-25",  # mismatched
            element_id="e1",
            doc_id="doc1",
            char_start=10,
            char_end=20,
        )


def test_element_offset_invariant() -> None:
    with pytest.raises(ValidationError):
        Element(
            element_id="e1",
            doc_id="d1",
            element_type=ElementType.PARAGRAPH,
            text="hi",
            page=1,
            char_start=100,
            char_end=50,  # invalid
            extraction_confidence=1.0,
        )


def test_verification_report_requires_all_four_gates() -> None:
    with pytest.raises(ValidationError):
        VerificationReport(
            claim_id="c1",
            gates=[
                GateResult(gate=GateName.SPAN_EXISTENCE, passed=True, ran_at=_now()),
            ],
            status=VerificationStatus.FLAGGED,
        )


def test_claim_round_trip() -> None:
    claim = Claim(
        claim_id="c1",
        indicator_id="6.4",
        pillar_id="6",
        clause_id="doc1_s26_p1",
        jurisdiction="SAMPLE",
        clause_pattern=ClausePattern.CONDITIONAL_REGIME,
        decomposition=Decomposition(
            subject="organisation",
            condition="transfer outside the country",
            constraint="recipient bound by comparable-protection obligations",
        ),
        evidence_spans=[
            EvidenceSpan(
                span_id="doc1#100-300",
                element_id="e26",
                doc_id="doc1",
                char_start=100,
                char_end=300,
            )
        ],
        regime=LegalRegime(primary_element_id="e26", member_element_ids=["e26", "e2"]),
        layer1_status=Layer1Status.PENDING_VERIFICATION,
        created_at=_now(),
    )
    redumped = Claim.model_validate_json(claim.model_dump_json())
    assert redumped == claim


def test_coverage_record_three_states() -> None:
    CoverageRecord(
        jurisdiction="SAMPLE",
        indicator_id="6.4",
        state=CoverageState.NO_EVIDENCE_IN_SEARCHED_CORPUS,
        measured_recall=0.82,
        reason="no clause matched",
    )
    CoverageRecord(
        jurisdiction="SAMPLE",
        indicator_id="6.4",
        state=CoverageState.EVIDENCE_FOUND,
        verified_claim_ids=["c1"],
    )


def test_document_meta_sha256_length() -> None:
    DocumentMeta(
        doc_id="d1",
        jurisdiction="SAMPLE",
        title="t",
        document_type=DocumentType.STATUTE,
        authority_tier=AuthorityTier.TIER_1_STATUTE,
        sha256="a" * 64,
        retrieved_at=_now(),
    )
    with pytest.raises(ValidationError):
        DocumentMeta(
            doc_id="d1",
            jurisdiction="SAMPLE",
            title="t",
            document_type=DocumentType.STATUTE,
            authority_tier=AuthorityTier.TIER_1_STATUTE,
            sha256="short",
            retrieved_at=_now(),
        )


def test_bounding_box_frozen() -> None:
    bb = BoundingBox(page=1, x0=0.0, y0=0.0, x1=1.0, y1=1.0)
    with pytest.raises(ValidationError):
        bb.page = 2  # type: ignore[misc]
