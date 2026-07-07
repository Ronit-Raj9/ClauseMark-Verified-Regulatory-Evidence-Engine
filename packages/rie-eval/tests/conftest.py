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


# Live gold under gold/ now carries the REAL Round-1 corpus (Singapore/Australia/
# Malaysia, dozens of P6/P7 rows). Unit tests of the recall/PRF *math* must not
# depend on that growing tree — they pin a frozen 3-item Pillar-7 gold set in a
# throwaway repo (only gold/ is needed; evaluate_pillar/measure_retrieval_recall
# touch nothing else).
_FROZEN_P7_GOLD = """\
items:
  - gold_id: "p07_frozen_consent_001"
    pillar_id: "7"
    indicator_id: "7.2"
    jurisdiction: "SAMPLE"
    doc_id: "sample_dpa_2020"
    span_text: "An organisation shall only process personal data with the consent of the data subject or on another lawful basis specified in section 6."
    expected_clause_pattern: "obligation"
    expected_score_band: "1"
    expected_authority_tier: "tier_1_statute"
    notes: "frozen fixture"
  - gold_id: "p07_frozen_breach_002"
    pillar_id: "7"
    indicator_id: "7.4"
    jurisdiction: "SAMPLE"
    doc_id: "sample_dpa_2020"
    span_text: "A controller shall notify the Commissioner without undue delay, and in any event within 72 hours, of any personal data breach, and shall notify affected data subjects where the breach is likely to result in a high risk to their rights."
    expected_clause_pattern: "obligation"
    expected_score_band: "1"
    expected_authority_tier: "tier_1_statute"
    notes: "frozen fixture"
  - gold_id: "p07_frozen_authority_003"
    pillar_id: "7"
    indicator_id: "7.5"
    jurisdiction: "SAMPLE"
    doc_id: "sample_dpa_2020"
    span_text: "The Data Protection Commission shall be independent in the exercise of its functions and shall not be subject to the direction or control of any person or authority."
    expected_clause_pattern: "obligation"
    expected_score_band: "1"
    expected_authority_tier: "tier_1_statute"
    notes: "frozen fixture"
"""


@pytest.fixture
def frozen_repo_root(tmp_path: Path) -> Path:
    """Throwaway repo with a pinned 3-item Pillar-7 gold set + real gold schema."""
    import shutil

    (tmp_path / "gold" / "_schema").mkdir(parents=True)
    shutil.copy(
        REPO_ROOT / "gold" / "_schema" / "gold_item.schema.json",
        tmp_path / "gold" / "_schema" / "gold_item.schema.json",
    )
    (tmp_path / "gold" / "pillar_07").mkdir(parents=True)
    (tmp_path / "gold" / "pillar_07" / "frozen.yaml").write_text(
        _FROZEN_P7_GOLD, encoding="utf-8"
    )
    return tmp_path


@pytest.fixture
def frozen_config(frozen_repo_root: Path) -> ConfigRepository:
    return ConfigRepository(repo_root=frozen_repo_root)


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
