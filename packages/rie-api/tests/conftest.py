"""Shared fixtures: in-memory repo + FastAPI TestClient with overridden deps.

Tests must NOT touch Postgres or Qdrant. The `app` fixture wires:
- an in-memory SQLite-backed `DocumentRepository` (real schema, fake engine)
- a real `ConfigRepository` pointing at the repo root (so pillar registry
  tests exercise the actual YAML), and
- a fake `RunGraph` callable that echoes the request back.

The lifespan is bypassed via `TestClient` not entering it — we attach state
manually on the app, mirroring what `main.lifespan` does in production.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from rie_api.deps import get_run_graph
from rie_api.main import create_app
from rie_api.schemas import RunRequest, RunResponse
from rie_api.settings import Settings
from rie_config.loader import ConfigRepository
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
    GateName,
    GateResult,
    Layer1Status,
    LegalRegime,
    VerificationReport,
    VerificationStatus,
)
from rie_persistence.models import Base
from rie_persistence.repository import DocumentRepository
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Path to the repository root (4 levels up: tests/ → rie-api → packages → root).
REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def settings() -> Settings:
    return Settings(
        use_in_memory_db=True,
        repo_root=REPO_ROOT,
        env="test",
        log_level="WARNING",
    )


@pytest.fixture
def doc_repo() -> DocumentRepository:
    engine = create_engine(
        "sqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return DocumentRepository(
        session_factory=sessionmaker(engine, future=True, expire_on_commit=False)
    )


@pytest.fixture
def config_repo(settings: Settings) -> ConfigRepository:
    return ConfigRepository(repo_root=settings.repo_root)


@pytest.fixture
def fake_run_graph() -> object:
    """Echo-style fake graph: returns a `RunResponse(accepted)` for any request."""

    def _graph(req: RunRequest) -> RunResponse:
        return RunResponse(
            run_id=req.run_id or "fake-run-id",
            status="accepted",
            detail=f"fake graph processed {len(req.pillar_ids)} pillar(s)",
        )

    return _graph


@pytest.fixture
def app(
    settings: Settings,
    doc_repo: DocumentRepository,
    config_repo: ConfigRepository,
    fake_run_graph: object,
) -> FastAPI:
    """Build a FastAPI app with deps wired by hand (lifespan is bypassed)."""
    # Build a fresh app — but DO NOT enter its lifespan; we attach state
    # manually so tests don't connect to Postgres.
    test_app = create_app()
    test_app.state.settings = settings
    test_app.state.doc_repo = doc_repo
    test_app.state.config_repo = config_repo
    test_app.state.run_graph = fake_run_graph
    # Override the dep so `get_run_graph` returns our fake without re-importing
    # rie_orchestration even on a cold app instance.
    test_app.dependency_overrides[get_run_graph] = lambda: fake_run_graph
    return test_app


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    # `with TestClient(app)` would enter the lifespan; we want to skip that to
    # avoid Postgres connection attempts. Instantiate without context.
    yield TestClient(app)


# ──────────────────────────────────────────────────────────────────────────
# Seed helpers used across tests
# ──────────────────────────────────────────────────────────────────────────


def _now() -> datetime:
    return datetime(2026, 5, 24, tzinfo=UTC)


@pytest.fixture
def seed_doc_and_element(doc_repo: DocumentRepository) -> tuple[str, str, str]:
    """Persist one document + one element. Returns (doc_id, element_id, text)."""
    text = "An organisation shall not transfer personal data outside the country."
    meta = DocumentMeta(
        doc_id="d1",
        jurisdiction="SAMPLE",
        title="Sample DPA",
        document_type=DocumentType.STATUTE,
        authority_tier=AuthorityTier.TIER_1_STATUTE,
        sha256="a" * 64,
        retrieved_at=_now(),
    )
    doc_repo.save_document(meta)
    el = Element(
        element_id="d1_s26_p1",
        doc_id="d1",
        element_type=ElementType.PARAGRAPH,
        text=text,
        page=1,
        char_start=0,
        char_end=len(text),
        extraction_confidence=0.99,
    )
    doc_repo.save_elements("d1", [el], [])
    return "d1", "d1_s26_p1", text


@pytest.fixture
def seed_claim(
    doc_repo: DocumentRepository,
    seed_doc_and_element: tuple[str, str, str],
) -> Claim:
    doc_id, element_id, text = seed_doc_and_element
    claim = Claim(
        claim_id="claim-001",
        indicator_id="6.4",
        pillar_id="6",
        clause_id=element_id,
        jurisdiction="SAMPLE",
        clause_pattern=ClausePattern.CONDITIONAL_REGIME,
        decomposition=Decomposition(
            subject="organisation",
            condition="cross-border transfer",
            constraint="comparable-protection requirement",
        ),
        evidence_spans=[
            EvidenceSpan(
                span_id=f"{doc_id}#0-{len(text)}",
                element_id=element_id,
                doc_id=doc_id,
                char_start=0,
                char_end=len(text),
            )
        ],
        regime=LegalRegime(primary_element_id=element_id, member_element_ids=[element_id]),
        layer1_status=Layer1Status.FLAGGED,
        created_at=_now(),
    )
    doc_repo.save_claim(claim)

    # Attach a complete (FLAGGED) verification report so the detail endpoint
    # returns something interesting.
    gates = [
        GateResult(gate=GateName.SPAN_EXISTENCE, passed=True, ran_at=_now()),
        GateResult(gate=GateName.VERBATIM_MATCH, passed=True, ran_at=_now()),
        GateResult(gate=GateName.ENTAILMENT, passed=False, detail="NLI disagreed", ran_at=_now()),
        GateResult(gate=GateName.SELF_CONSISTENCY, passed=True, ran_at=_now()),
    ]
    report = VerificationReport(
        claim_id=claim.claim_id,
        gates=gates,
        status=VerificationStatus.FLAGGED,
        failure_reasons=["entailment disagreement"],
    )
    doc_repo.save_verification(report)
    return claim


@pytest.fixture
def seed_coverage(doc_repo: DocumentRepository) -> CoverageRecord:
    rec = CoverageRecord(
        jurisdiction="SAMPLE",
        indicator_id="6.4",
        state=CoverageState.NO_EVIDENCE_IN_SEARCHED_CORPUS,
        measured_recall=0.82,
        reason="no clauses passed all gates",
    )
    doc_repo.save_coverage(rec)
    return rec
