"""Shared fixtures + tiny in-memory fakes for `Evaluator` tests."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pytest
from rie_config import ConfigRepository
from rie_contracts import (
    AuthorityTier,
    Claim,
    ClausePattern,
    CoverageRecord,
    CoverageState,
    Decomposition,
    DocumentMeta,
    DocumentType,
    EvidenceSpan,
    Layer1Status,
    LegalRegime,
    ReviewDecision,
    ReviewRecord,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def config(repo_root: Path) -> ConfigRepository:
    return ConfigRepository(repo_root=repo_root)


@dataclass
class FakeRepo:
    """Minimal in-memory stand-in for the bits of `DocumentRepositoryPort`
    that the Evaluator actually reads. Tests use this — no Postgres needed.
    """

    documents: dict[str, DocumentMeta] = field(default_factory=dict)
    reviews: dict[str, list[ReviewRecord]] = field(default_factory=dict)
    coverage: list[CoverageRecord] = field(default_factory=list)

    def get_document(self, doc_id: str) -> DocumentMeta:
        if doc_id not in self.documents:
            raise KeyError(doc_id)
        return self.documents[doc_id]

    def list_reviews(self, claim_id: str) -> Sequence[ReviewRecord]:
        return list(self.reviews.get(claim_id, []))

    def list_coverage(self, jurisdiction: str | None = None) -> Sequence[CoverageRecord]:
        if jurisdiction is None:
            return list(self.coverage)
        return [c for c in self.coverage if c.jurisdiction == jurisdiction]


@pytest.fixture
def fake_repo() -> FakeRepo:
    return FakeRepo()


@pytest.fixture
def now() -> datetime:
    return datetime(2026, 5, 24, tzinfo=UTC)


def make_claim(
    *,
    claim_id: str,
    indicator_id: str,
    pillar_id: str,
    doc_id: str = "sample_dpa_2020",
    jurisdiction: str = "SAMPLE",
    clause_pattern: ClausePattern = ClausePattern.OBLIGATION,
    char_start: int = 100,
    char_end: int | None = None,
    status: Layer1Status = Layer1Status.VERIFIED,
    created_at: datetime | None = None,
) -> Claim:
    """Factory: build a synthetic Claim with a single primary evidence span."""
    ts = created_at or datetime(2026, 5, 24, tzinfo=UTC)
    if char_end is None:
        char_end = char_start + 100
    element_id = f"{doc_id}_elem_{char_start}"
    return Claim(
        claim_id=claim_id,
        indicator_id=indicator_id,
        pillar_id=pillar_id,
        clause_id=element_id,
        jurisdiction=jurisdiction,
        clause_pattern=clause_pattern,
        decomposition=Decomposition(
            subject="organisation",
            condition=None,
            constraint="comply with the obligation",
        ),
        evidence_spans=[
            EvidenceSpan(
                span_id=f"{doc_id}#{char_start}-{char_end}",
                element_id=element_id,
                doc_id=doc_id,
                char_start=char_start,
                char_end=char_end,
            )
        ],
        regime=LegalRegime(primary_element_id=element_id, member_element_ids=[element_id]),
        layer1_status=status,
        created_at=ts,
    )


def make_doc_meta(
    *,
    doc_id: str = "sample_dpa_2020",
    jurisdiction: str = "SAMPLE",
    authority_tier: AuthorityTier = AuthorityTier.TIER_1_STATUTE,
    now: datetime | None = None,
) -> DocumentMeta:
    ts = now or datetime(2026, 5, 24, tzinfo=UTC)
    return DocumentMeta(
        doc_id=doc_id,
        jurisdiction=jurisdiction,
        title="Sample DPA",
        document_type=DocumentType.STATUTE,
        authority_tier=authority_tier,
        sha256="a" * 64,
        retrieved_at=ts,
    )


def make_review(
    *,
    claim_id: str,
    decision: ReviewDecision,
    at: datetime | None = None,
) -> ReviewRecord:
    return ReviewRecord(
        claim_id=claim_id,
        reviewer="lead_reviewer",
        decision=decision,
        note="",
        decided_at=at or datetime(2026, 5, 24, tzinfo=UTC),
    )


def make_coverage(
    *,
    jurisdiction: str,
    indicator_id: str,
    state: CoverageState,
    measured_recall: float | None = None,
) -> CoverageRecord:
    return CoverageRecord(
        jurisdiction=jurisdiction,
        indicator_id=indicator_id,
        state=state,
        measured_recall=measured_recall,
    )
