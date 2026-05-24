"""End-to-end integration test on data/samples/ with the fake adapter bundle.

No Postgres / Qdrant / LLM required. Runs through ingest → extract → index →
retrieve → classify → verify → coverage and asserts the evidence package shape.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from rie_orchestration import run_pipeline

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.integration
def test_sample_run_produces_evidence_package() -> None:
    package = run_pipeline(
        jurisdiction="SAMPLE",
        pillar_ids=["6", "7"],
        use_fakes=True,
        repo_root=REPO_ROOT,
    )
    assert package["run_id"]
    assert package["counts"]["documents"] >= 1
    assert package["counts"]["claims"] >= 1
    # FakeVerifier passes all gates → some claims should be VERIFIED.
    assert package["counts"]["verified"] >= 1
    states = {row["state"] for row in package["coverage"]}
    assert "evidence_found" in states


@pytest.mark.integration
def test_run_jurisdiction_consistent_in_claims() -> None:
    package = run_pipeline(
        jurisdiction="SAMPLE",
        pillar_ids=["6", "7"],
        use_fakes=True,
        repo_root=REPO_ROOT,
    )
    for claim in package["claims"]:
        assert claim["jurisdiction"] == "SAMPLE"


@pytest.mark.integration
def test_three_state_absence_never_emits_bare_zero() -> None:
    package = run_pipeline(
        jurisdiction="SAMPLE",
        pillar_ids=["6", "7"],
        use_fakes=True,
        repo_root=REPO_ROOT,
    )
    valid_states = {
        "evidence_found",
        "no_evidence_in_searched_corpus",
        "insufficient_coverage",
    }
    for row in package["coverage"]:
        assert row["state"] in valid_states
