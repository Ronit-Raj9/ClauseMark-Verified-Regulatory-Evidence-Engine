"""Persistence smoke tests using in-memory SQLite (Postgres-only JSONB shim).

These tests verify mapping correctness without requiring a running Postgres.
Real Postgres is covered by `tests/integration/`.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from rie_contracts import (
    AuthorityTier,
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
    Layer1Status,
    LegalRegime,
)
from rie_persistence.models import Base
from rie_persistence.repository import DocumentRepository
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture
def repo() -> DocumentRepository:
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    return DocumentRepository(
        session_factory=sessionmaker(engine, future=True, expire_on_commit=False)
    )


def _now() -> datetime:
    return datetime(2026, 5, 24, tzinfo=UTC)


def test_save_document_and_element_roundtrip(repo: DocumentRepository) -> None:
    meta = DocumentMeta(
        doc_id="d1",
        jurisdiction="SAMPLE",
        title="Test",
        document_type=DocumentType.STATUTE,
        authority_tier=AuthorityTier.TIER_1_STATUTE,
        sha256="a" * 64,
        retrieved_at=_now(),
    )
    repo.save_document(meta)
    fetched = repo.get_document("d1")
    assert fetched.title == "Test"

    e = Element(
        element_id="d1_s26_p1",
        doc_id="d1",
        element_type=ElementType.PARAGRAPH,
        text="hello world",
        page=1,
        char_start=0,
        char_end=11,
        extraction_confidence=0.99,
    )
    repo.save_elements("d1", [e], [])
    assert repo.get_element_text("d1_s26_p1") == "hello world"


def test_save_claim_roundtrip(repo: DocumentRepository) -> None:
    meta = DocumentMeta(
        doc_id="d1",
        jurisdiction="SAMPLE",
        title="Test",
        document_type=DocumentType.STATUTE,
        authority_tier=AuthorityTier.TIER_1_STATUTE,
        sha256="b" * 64,
        retrieved_at=_now(),
    )
    repo.save_document(meta)
    e = Element(
        element_id="d1_s26",
        doc_id="d1",
        element_type=ElementType.SECTION,
        text="hi",
        page=1,
        char_start=0,
        char_end=2,
        extraction_confidence=1.0,
    )
    repo.save_elements("d1", [e], [])
    c = Claim(
        claim_id="c1",
        indicator_id="6.4",
        pillar_id="6",
        clause_id="d1_s26",
        jurisdiction="SAMPLE",
        clause_pattern=ClausePattern.CONDITIONAL_REGIME,
        decomposition=Decomposition(subject="x", constraint="y"),
        evidence_spans=[
            EvidenceSpan(
                span_id="d1#0-2",
                element_id="d1_s26",
                doc_id="d1",
                char_start=0,
                char_end=2,
            )
        ],
        regime=LegalRegime(primary_element_id="d1_s26", member_element_ids=["d1_s26"]),
        layer1_status=Layer1Status.PENDING_VERIFICATION,
        created_at=_now(),
    )
    repo.save_claim(c)
    got = repo.get_claim("c1")
    assert got.indicator_id == "6.4"


def test_layer2_recommendation_roundtrip(repo: DocumentRepository) -> None:
    from rie_contracts import Layer2Recommendation, ScoreBand

    meta = DocumentMeta(
        doc_id="d1",
        jurisdiction="SAMPLE",
        title="Test",
        document_type=DocumentType.STATUTE,
        authority_tier=AuthorityTier.TIER_1_STATUTE,
        sha256="c" * 64,
        retrieved_at=_now(),
    )
    repo.save_document(meta)
    c = Claim(
        claim_id="c2",
        indicator_id="6.4",
        pillar_id="6",
        clause_id="d1_s26",
        jurisdiction="SAMPLE",
        clause_pattern=ClausePattern.CONDITIONAL_REGIME,
        decomposition=Decomposition(subject="x", constraint="y"),
        evidence_spans=[],
        regime=LegalRegime(primary_element_id="d1_s26", member_element_ids=["d1_s26"]),
        layer1_status=Layer1Status.VERIFIED,
        created_at=_now(),
    )
    repo.save_claim(c)
    rec = Layer2Recommendation(
        claim_id="c2",
        indicator_id="6.4",
        recommended_band=ScoreBand.HALF,
        rationale="test band",
        open_questions=["breadth unknown"],
    )
    repo.save_layer2_recommendation(rec)
    loaded = repo.get_layer2_recommendation("c2")
    assert loaded is not None
    assert loaded.recommended_band == ScoreBand.HALF
    assert loaded.open_questions == ["breadth unknown"]


def test_layer2_survives_claim_resave(repo: DocumentRepository) -> None:
    from rie_contracts import Layer2Recommendation, ScoreBand

    c = Claim(
        claim_id="c3",
        indicator_id="6.4",
        pillar_id="6",
        clause_id="d1_s26",
        jurisdiction="SAMPLE",
        clause_pattern=ClausePattern.CONDITIONAL_REGIME,
        decomposition=Decomposition(subject="x", constraint="y"),
        evidence_spans=[],
        regime=LegalRegime(primary_element_id="d1_s26", member_element_ids=["d1_s26"]),
        layer1_status=Layer1Status.VERIFIED,
        created_at=_now(),
    )
    repo.save_claim(c)
    repo.save_layer2_recommendation(
        Layer2Recommendation(
            claim_id="c3",
            indicator_id="6.4",
            recommended_band=ScoreBand.ONE,
            rationale="kept",
        )
    )
    repo.save_claim(c)
    loaded = repo.get_layer2_recommendation("c3")
    assert loaded is not None
    assert loaded.recommended_band == ScoreBand.ONE


def test_list_layer2_recommendations_by_jurisdiction(repo: DocumentRepository) -> None:
    from rie_contracts import Layer2Recommendation, ScoreBand

    for claim_id, jurisdiction in (("c4", "SAMPLE"), ("c5", "OTHER")):
        repo.save_claim(
            Claim(
                claim_id=claim_id,
                indicator_id="6.4",
                pillar_id="6",
                clause_id="d1_s26",
                jurisdiction=jurisdiction,
                clause_pattern=ClausePattern.CONDITIONAL_REGIME,
                decomposition=Decomposition(subject="x", constraint="y"),
                evidence_spans=[],
                regime=LegalRegime(primary_element_id="d1_s26", member_element_ids=["d1_s26"]),
                layer1_status=Layer1Status.VERIFIED,
                created_at=_now(),
            )
        )
        repo.save_layer2_recommendation(
            Layer2Recommendation(
                claim_id=claim_id,
                indicator_id="6.4",
                recommended_band=ScoreBand.HALF,
                rationale=f"band for {claim_id}",
            )
        )

    sample = list(repo.list_layer2_recommendations(jurisdiction="SAMPLE"))
    assert len(sample) == 1
    assert sample[0].claim_id == "c4"

    pillar6 = list(repo.list_layer2_recommendations(pillar_id="6"))
    assert len(pillar6) == 2


def test_get_layer2_recommendation_missing_claim_raises(repo: DocumentRepository) -> None:
    import pytest

    with pytest.raises(KeyError):
        repo.get_layer2_recommendation("missing-claim")


def test_coverage_upsert(repo: DocumentRepository) -> None:
    rec = CoverageRecord(
        jurisdiction="SAMPLE",
        indicator_id="6.4",
        state=CoverageState.NO_EVIDENCE_IN_SEARCHED_CORPUS,
        measured_recall=0.82,
        reason="none found",
    )
    repo.save_coverage(rec)
    rec2 = CoverageRecord(
        jurisdiction="SAMPLE",
        indicator_id="6.4",
        state=CoverageState.EVIDENCE_FOUND,
        verified_claim_ids=["c1"],
    )
    repo.save_coverage(rec2)
    rows = list(repo.list_coverage("SAMPLE"))
    assert len(rows) == 1
    assert rows[0].state == CoverageState.EVIDENCE_FOUND
