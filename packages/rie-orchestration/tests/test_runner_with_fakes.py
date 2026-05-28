"""End-to-end run on `data/samples/sample_dpa.txt` using fake adapters.

Verifies that the orchestration graph wires together correctly without
requiring Postgres, Qdrant, or any LLM.
"""

from __future__ import annotations

from pathlib import Path

from rie_config import ConfigRepository
from rie_orchestration import run_pipeline
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


def test_full_pipeline_on_sample() -> None:
    package = run_pipeline(
        jurisdiction="SAMPLE",
        pillar_ids=["6", "7"],
        bundle=_make_bundle(),
        enable_hitl=False,
    )
    assert package["counts"]["documents"] >= 1
    assert package["counts"]["claims"] > 0
    assert package["counts"]["verified"] > 0
    coverage_states = {c["state"] for c in package["coverage"]}
    # At least some indicators have evidence_found
    assert "evidence_found" in coverage_states
    assert package.get("layer2_recommendations") is not None
    if package["counts"]["verified"] > 0:
        assert len(package["layer2_recommendations"]) >= 1
        assert package["layer2_recommendations"][0]["human_confirmation_required"] is True


def test_run_produces_jurisdiction_consistent_claims() -> None:
    package = run_pipeline(
        jurisdiction="SAMPLE",
        pillar_ids=["6", "7"],
        bundle=_make_bundle(),
        enable_hitl=False,
    )
    for claim in package["claims"]:
        assert claim["jurisdiction"] == "SAMPLE"


def test_uses_only_built_pillars() -> None:
    package = run_pipeline(
        jurisdiction="SAMPLE",
        pillar_ids=["6", "7", "8"],
        bundle=_make_bundle(),
        enable_hitl=False,
    )
    # Pillar 8 is built in Phase 2 — retrieve runs and coverage rows are emitted.
    pillar_8_indicators = {
        c["indicator_id"] for c in package["coverage"] if c["indicator_id"].startswith("8.")
    }
    assert pillar_8_indicators, "built pillar 8 should produce coverage rows"
    legal_states = {
        "evidence_found",
        "insufficient_coverage",
        "no_evidence_in_searched_corpus",
    }
    for c in package["coverage"]:
        if c["indicator_id"].startswith("8."):
            assert c["state"] in legal_states
