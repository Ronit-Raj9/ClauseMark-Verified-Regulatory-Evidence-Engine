"""build_run_graph + resume_pipeline API surface tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from rie_contracts import Layer1Status, VerificationStatus
from rie_orchestration import build_run_graph, resume_pipeline, run_pipeline
from rie_orchestration.fakes import (
    FakeClassifier,
    FakeCoverage,
    FakeExtractor,
    FakeIngest,
    FakeRepo,
    FakeRetrieval,
    FakeVerifier,
)
from rie_orchestration.wiring import AdapterBundle

REPO_ROOT = Path(__file__).resolve().parents[3]


def _make_bundle() -> AdapterBundle:
    from rie_config import ConfigRepository

    config = ConfigRepository(repo_root=REPO_ROOT)
    repo = FakeRepo()
    return AdapterBundle(
        config=config,
        ingest=FakeIngest(config),
        extractor=FakeExtractor(),
        retrieval=FakeRetrieval(),
        classifier=FakeClassifier(get_element=repo.get_element),
        verifier=FakeVerifier(),
        coverage=FakeCoverage(),
        repo=repo,
        samples_dir=REPO_ROOT / "data" / "samples",
    )


class _Req:
    def __init__(self, jurisdiction: str, pillar_ids: list[str], run_id: str | None = None):
        self.jurisdiction = jurisdiction
        self.pillar_ids = pillar_ids
        self.run_id = run_id


def test_build_run_graph_returns_callable() -> None:
    run_graph = build_run_graph(repo_root=REPO_ROOT, use_fakes=True)
    resp = run_graph(_Req("SAMPLE", ["6", "7"], run_id="api-run-1"))
    assert resp["run_id"] == "api-run-1"
    assert resp["status"] in {"completed", "running", "failed"}


def test_run_pipeline_records_node_spans() -> None:
    package = run_pipeline(
        jurisdiction="SAMPLE",
        pillar_ids=["6"],
        bundle=_make_bundle(),
        enable_hitl=False,
    )
    events = {e["name"] for e in package["tracing"]["events"]}
    assert "ingest" in events
    assert "coverage" in events


def test_evidence_package_includes_citations() -> None:
    package = run_pipeline(
        jurisdiction="SAMPLE",
        pillar_ids=["6"],
        bundle=_make_bundle(),
        enable_hitl=False,
    )
    assert package.get("citations")
    first_claim_id = package["claims"][0]["claim_id"]
    assert first_claim_id in package["citations"]


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("langgraph") is None,
    reason="langgraph required",
)
def test_resume_pipeline_accepts_flagged_claims() -> None:
    from dataclasses import dataclass
    from datetime import UTC, datetime

    from rie_contracts import GateName, GateResult, VerificationReport
    from rie_orchestration.graph import build_memory_checkpointer

    build_memory_checkpointer(reset=True)

    @dataclass
    class FlaggingVerifier:
        def verify(self, claim, get_element_text):
            del get_element_text
            gates = [GateResult(gate=g, passed=True, ran_at=datetime.now(UTC)) for g in GateName]
            return VerificationReport(
                claim_id=claim.claim_id,
                gates=gates,
                status=VerificationStatus.FLAGGED,
            )

    bundle = _make_bundle()
    bundle = AdapterBundle(
        config=bundle.config,
        ingest=bundle.ingest,
        extractor=bundle.extractor,
        retrieval=bundle.retrieval,
        classifier=bundle.classifier,
        verifier=FlaggingVerifier(),
        coverage=bundle.coverage,
        repo=bundle.repo,
        samples_dir=bundle.samples_dir,
    )
    run_id = "resume-test-1"
    package = run_pipeline(
        jurisdiction="SAMPLE",
        pillar_ids=["6"],
        bundle=bundle,
        run_id=run_id,
        enable_hitl=True,
        enable_postgres_checkpointer=False,
    )
    assert package["status"] == "running"
    flagged_ids = [
        cid
        for cid, rep in package["verifications"].items()
        if rep["status"] == VerificationStatus.FLAGGED.value
    ]
    decisions = dict.fromkeys(flagged_ids, "accept")
    resumed = resume_pipeline(
        run_id,
        decisions,
        bundle=bundle,
        enable_postgres_checkpointer=False,
    )
    assert resumed["status"] == "completed"
    for cid in flagged_ids:
        claim = next(c for c in resumed["claims"] if c["claim_id"] == cid)
        assert claim["layer1_status"] == Layer1Status.VERIFIED.value
