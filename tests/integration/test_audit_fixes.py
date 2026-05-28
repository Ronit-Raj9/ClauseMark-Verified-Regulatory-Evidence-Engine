"""Integration tests for §4–§7 audit fixes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[2]


class _RunRequest:
    """Minimal stand-in for API ``RunRequest`` in ``build_run_graph`` tests."""

    def __init__(
        self,
        jurisdiction: str,
        pillar_ids: list[str],
        run_id: str | None = None,
    ) -> None:
        self.jurisdiction = jurisdiction
        self.pillar_ids = pillar_ids
        self.run_id = run_id


@pytest.mark.integration
def test_build_run_graph_callable_returns_run_id_and_status() -> None:
    """``build_run_graph`` is the API ``RunGraph`` factory (§4 wiring)."""
    from rie_orchestration import build_run_graph

    run_graph = build_run_graph(repo_root=REPO_ROOT, use_fakes=True)
    resp = run_graph(_RunRequest("SAMPLE", ["6"], run_id="api-run-graph-1"))
    assert resp["run_id"] == "api-run-graph-1"
    assert resp["status"] in {"completed", "running", "failed"}
    assert resp["detail"]


@pytest.mark.integration
def test_api_post_runs_uses_build_run_graph_path() -> None:
    """``POST /v1/runs`` invokes the wired ``RunGraph`` callable end-to-end."""
    from rie_api.deps import get_run_graph
    from rie_api.main import create_app
    from rie_orchestration import build_run_graph

    run_graph = build_run_graph(repo_root=REPO_ROOT, use_fakes=True)
    app = create_app()
    app.state.run_graph = run_graph
    app.dependency_overrides[get_run_graph] = lambda: run_graph

    client = TestClient(app, raise_server_exceptions=False)
    response = client.post(
        "/v1/runs",
        json={"jurisdiction": "SAMPLE", "pillar_ids": ["6"], "run_id": "api-post-1"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["run_id"] == "api-post-1"
    assert body["status"] in {"completed", "running", "failed", "accepted"}
    assert body["detail"]


@pytest.mark.integration
def test_pipeline_layer2_recommendations_are_human_gated() -> None:
    """Layer-2 output is a recommendation bundle, never a final score (§6)."""
    from rie_orchestration import run_pipeline

    package = run_pipeline(
        jurisdiction="SAMPLE",
        pillar_ids=["6"],
        use_fakes=True,
        repo_root=REPO_ROOT,
        enable_hitl=False,
    )
    recs = package.get("layer2_recommendations", [])
    assert isinstance(recs, list)
    assert recs, "expected Layer-2 recommendations for verified claims"
    for rec in recs:
        assert rec["human_confirmation_required"] is True
        assert rec["recommended_band"] in {"0", "0.5", "1", "null"}
        assert rec["claim_id"]
        assert rec["indicator_id"]
        assert rec["rationale"]


@pytest.mark.integration
def test_confidence_screening_in_verify_path() -> None:
    from rie_orchestration import run_pipeline

    package = run_pipeline(
        jurisdiction="SAMPLE",
        pillar_ids=["6"],
        use_fakes=True,
        repo_root=REPO_ROOT,
        enable_hitl=False,
    )
    assert package["run_id"]


@pytest.mark.integration
def test_gold_recall_is_measured_not_placeholder() -> None:
    """Gold-set recall comes from hit overlap, not a hard-coded 0.82 stub (§7)."""
    from rie_contracts import RetrievalHit
    from rie_orchestration.gold_recall import lookup_gold_recall
    from rie_orchestration.wiring import build_default_bundle

    bundle = build_default_bundle(REPO_ROOT, use_fakes=True)
    state = {
        "run_id": "recall-test",
        "jurisdiction": "SAMPLE",
        "pillar_ids": ["6"],
        "documents": [],
        "raw_bytes_by_doc": {},
        "elements_by_doc": {},
        "edges_by_doc": {},
        "candidate_hits_by_pillar": {
            "6": [
                RetrievalHit(
                    element_id="sample_dpa_2020_s26_chunk0",
                    doc_id="sample_dpa_2020",
                    parent_element_id="sample_dpa_2020_s26",
                    neighbourhood_element_ids=[],
                    score=0.9,
                    snippet="transfer personal data outside the country",
                )
            ]
        },
        "claims": [],
        "verifications": {},
        "coverage": [],
        "errors": [],
    }

    recall = lookup_gold_recall(bundle, "6", "6.4", state=state)
    assert recall is not None
    assert recall == 1.0
    assert recall != 0.82

    state["ingest_degraded"] = True
    assert lookup_gold_recall(bundle, "6", "6.4", state=state) is None


@pytest.mark.integration
def test_no_evidence_coverage_carries_measured_recall() -> None:
    """Coverage ``no_evidence`` rows attach computed recall, not a placeholder."""
    from rie_contracts import CoverageState, RetrievalHit
    from rie_orchestration.fakes import FakeCoverage
    from rie_orchestration.gold_recall import lookup_gold_recall
    from rie_orchestration.nodes import coverage_node
    from rie_orchestration.wiring import AdapterBundle, build_default_bundle

    bundle = build_default_bundle(REPO_ROOT, use_fakes=True)
    bundle = AdapterBundle(
        config=bundle.config,
        ingest=bundle.ingest,
        extractor=bundle.extractor,
        retrieval=bundle.retrieval,
        classifier=bundle.classifier,
        verifier=bundle.verifier,
        coverage=FakeCoverage(),
        repo=bundle.repo,
        samples_dir=bundle.samples_dir,
    )
    state = {
        "run_id": "coverage-recall",
        "jurisdiction": "SAMPLE",
        "pillar_ids": ["6"],
        "documents": [],
        "raw_bytes_by_doc": {},
        "elements_by_doc": {},
        "edges_by_doc": {},
        "candidate_hits_by_pillar": {
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
        },
        "claims": [],
        "verifications": {},
        "coverage": [],
        "errors": [],
    }
    expected_recall = lookup_gold_recall(bundle, "6", "6.4", state=state)
    assert expected_recall is not None
    assert expected_recall != 0.82

    out = coverage_node(state, bundle)
    row = next(r for r in out["coverage"] if r.indicator_id == "6.4")
    assert row.state == CoverageState.NO_EVIDENCE_IN_SEARCHED_CORPUS
    assert row.measured_recall == expected_recall
    assert row.measured_recall != 0.82


@pytest.mark.integration
def test_api_resume_endpoint_contract() -> None:
    """Resume returns structured response even when no interrupt pending."""
    from rie_api.deps import get_settings
    from rie_api.main import create_app
    from rie_api.settings import Settings

    app = create_app()
    settings = Settings(use_in_memory_db=True, repo_root=REPO_ROOT, env="test")
    app.state.settings = settings
    app.dependency_overrides[get_settings] = lambda: settings

    client = TestClient(app, raise_server_exceptions=False)
    response = client.post(
        "/v1/runs/fake-run/resume",
        json={"decisions": {"claim-001": "accept"}},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["run_id"] == "fake-run"
    assert body["status"] in {"completed", "failed", "interrupted", "unknown"}


@pytest.mark.integration
@pytest.mark.skipif(
    __import__("importlib").util.find_spec("langgraph") is None,
    reason="langgraph required for HITL resume",
)
def test_hitl_resume_accepts_flagged_claims() -> None:
    """HITL interrupt → ``resume_pipeline`` completes flagged claims (§4)."""
    from rie_contracts import GateName, GateResult, Layer1Status, VerificationReport, VerificationStatus
    from rie_orchestration import resume_pipeline, run_pipeline
    from rie_orchestration.fakes import (
        FakeClassifier,
        FakeCoverage,
        FakeExtractor,
        FakeIngest,
        FakeRepo,
        FakeRetrieval,
    )
    from rie_orchestration.graph import build_memory_checkpointer
    from rie_orchestration.wiring import AdapterBundle, build_default_bundle

    build_memory_checkpointer(reset=True)

    @dataclass
    class _FlaggingVerifier:
        def verify(self, claim, get_element_text):
            del get_element_text
            gates = [
                GateResult(gate=g, passed=True, ran_at=datetime.now(UTC)) for g in GateName
            ]
            return VerificationReport(
                claim_id=claim.claim_id,
                gates=gates,
                status=VerificationStatus.FLAGGED,
            )

    base = build_default_bundle(REPO_ROOT, use_fakes=True)
    repo = FakeRepo()
    bundle = AdapterBundle(
        config=base.config,
        ingest=FakeIngest(base.config),
        extractor=FakeExtractor(),
        retrieval=FakeRetrieval(),
        classifier=FakeClassifier(get_element=repo.get_element),
        verifier=_FlaggingVerifier(),
        coverage=FakeCoverage(),
        repo=repo,
        samples_dir=REPO_ROOT / "data" / "samples",
    )
    run_id = "hitl-resume-integration"
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
    assert flagged_ids

    resumed = resume_pipeline(
        run_id,
        {cid: "accept" for cid in flagged_ids},
        bundle=bundle,
        enable_postgres_checkpointer=False,
    )
    assert resumed["status"] == "completed"
    for cid in flagged_ids:
        claim = next(c for c in resumed["claims"] if c["claim_id"] == cid)
        assert claim["layer1_status"] == Layer1Status.VERIFIED.value
