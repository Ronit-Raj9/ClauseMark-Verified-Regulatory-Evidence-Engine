"""Tests for gold recall lookup and ingest graceful degradation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from rie_config import ConfigRepository
from rie_contracts import (
    Claim,
    CoverageRecord,
    CoverageState,
    DocumentMeta,
    DocumentType,
    SourceRegistryEntry,
)
from rie_orchestration.fakes import FakeRepo
from rie_orchestration.gold_recall import lookup_gold_recall
from rie_orchestration.nodes import coverage_node, ingest_node
from rie_orchestration.state import RieState
from rie_orchestration.wiring import AdapterBundle

REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class _FailingIngest:
    config: ConfigRepository
    fail_registry: bool = False
    fail_all_sources: bool = False

    def load_source_registry(self, jurisdiction: str) -> Sequence[SourceRegistryEntry]:
        del jurisdiction
        if self.fail_registry:
            from rie_ingest import IngestError

            raise IngestError("registry unreachable")
        return list(self.config.load_source_registry("SAMPLE"))

    def load_document_bytes(self, entry: SourceRegistryEntry) -> tuple[DocumentMeta, bytes]:
        if self.fail_all_sources:
            from rie_ingest import IngestError

            raise IngestError(f"GET {entry.source_id} -> 503")
        path = REPO_ROOT / entry.local_path  # type: ignore[operator]
        raw = path.read_bytes()
        meta = DocumentMeta(
            doc_id=entry.source_id,
            jurisdiction=entry.jurisdiction,
            title=entry.source_id,
            source_url=entry.source_url or "",
            document_type=DocumentType.STATUTE,
            authority_tier=entry.authority_tier,
            effective_date=entry.effective_date,
            sha256_hash="fake",
        )
        return meta, raw


@dataclass
class _StubCoverage:
    records: list[CoverageRecord] = field(default_factory=list)

    def evaluate(
        self,
        jurisdiction: str,
        indicator_id: str,
        verified_claims: Sequence[Claim],
        gold_recall: float | None,
    ) -> CoverageRecord:
        del verified_claims
        rec = CoverageRecord(
            jurisdiction=jurisdiction,
            indicator_id=indicator_id,
            state=(
                CoverageState.INSUFFICIENT_COVERAGE
                if gold_recall is None
                else CoverageState.NO_EVIDENCE_IN_SEARCHED_CORPUS
            ),
            measured_recall=gold_recall,
            reason=None if gold_recall is not None else "no recall measured",
        )
        self.records.append(rec)
        return rec


def _base_state() -> RieState:
    return {
        "run_id": "test",
        "jurisdiction": "SAMPLE",
        "pillar_ids": ["6"],
        "documents": [],
        "raw_bytes_by_doc": {},
        "elements_by_doc": {},
        "edges_by_doc": {},
        "candidate_hits_by_pillar": {},
        "claims": [],
        "verifications": {},
        "coverage": [],
        "errors": [],
    }


def _make_bundle(ingest: _FailingIngest, coverage: _StubCoverage) -> AdapterBundle:
    repo = FakeRepo()
    from rie_orchestration.fakes import (
        FakeClassifier,
        FakeExtractor,
        FakeRetrieval,
        FakeVerifier,
    )

    return AdapterBundle(
        config=ingest.config,
        ingest=ingest,
        extractor=FakeExtractor(),
        retrieval=FakeRetrieval(),
        classifier=FakeClassifier(get_element=repo.get_element),
        verifier=FakeVerifier(),
        coverage=coverage,
        repo=repo,
        samples_dir=REPO_ROOT / "data" / "samples",
    )


def test_ingest_registry_failure_sets_degraded_without_crash() -> None:
    config = ConfigRepository(repo_root=REPO_ROOT)
    ingest = _FailingIngest(config, fail_registry=True)
    bundle = _make_bundle(ingest, _StubCoverage())
    state = _base_state()
    out = ingest_node(state, bundle)
    assert out.get("ingest_degraded") is True
    assert out["documents"] == []
    assert any("ingest registry" in e for e in out.get("errors", []))


def test_ingest_all_sources_fail_sets_degraded() -> None:
    config = ConfigRepository(repo_root=REPO_ROOT)
    ingest = _FailingIngest(config, fail_all_sources=True)
    bundle = _make_bundle(ingest, _StubCoverage())
    state = _base_state()
    out = ingest_node(state, bundle)
    assert out.get("ingest_degraded") is True
    assert out["documents"] == []


def test_coverage_uses_insufficient_when_ingest_degraded() -> None:
    config = ConfigRepository(repo_root=REPO_ROOT)
    coverage = _StubCoverage()
    bundle = _make_bundle(_FailingIngest(config), coverage)
    state = _base_state()
    state["ingest_degraded"] = True
    state["candidate_hits_by_pillar"] = {"6": []}
    out = coverage_node(state, bundle)
    assert out["coverage"]
    assert all(r.state == CoverageState.INSUFFICIENT_COVERAGE for r in out["coverage"])


def test_gold_recall_none_when_ingest_degraded() -> None:
    config = ConfigRepository(repo_root=REPO_ROOT)
    bundle = _make_bundle(_FailingIngest(config), _StubCoverage())
    state = _base_state()
    state["ingest_degraded"] = True
    assert lookup_gold_recall(bundle, "6", "6.4", state=state) is None


def test_gold_recall_from_run_hits_when_gold_present() -> None:
    config = ConfigRepository(repo_root=REPO_ROOT)
    bundle = _make_bundle(_FailingIngest(config), _StubCoverage())
    state = _base_state()
    from rie_contracts import RetrievalHit

    state["candidate_hits_by_pillar"] = {
        "6": [
            RetrievalHit(
                element_id="sample_dpa_2020_s26_chunk0",
                doc_id="sample_dpa_2020",
                parent_element_id="sample_dpa_2020_s26",
                neighbourhood_element_ids=[],
                score=0.9,
                snippet="transfer personal data outside",
            )
        ]
    }
    recall = lookup_gold_recall(bundle, "6", "6.4", state=state)
    assert recall is not None
    assert 0.0 <= recall <= 1.0
