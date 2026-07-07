"""CLI tests — drive the Typer app via its test runner."""

from __future__ import annotations

from pathlib import Path

import pytest
from rie_eval.cli import app
from typer.testing import CliRunner

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_gold_validate_succeeds_on_real_gold(runner: CliRunner) -> None:
    """`gold validate` must pass against the real `gold/pillar_06`/`pillar_07`."""
    result = runner.invoke(app, ["gold", "validate", "--repo-root", str(REPO_ROOT)])
    assert result.exit_code == 0, result.output
    assert "pillar 6" in result.output
    assert "pillar 7" in result.output
    assert "FAIL" not in result.output


def test_gold_list_specific_pillar(runner: CliRunner) -> None:
    result = runner.invoke(app, ["gold", "list", "--pillar", "6", "--repo-root", str(REPO_ROOT)])
    assert result.exit_code == 0, result.output
    # Pillar 6 has at least the one canonical 6.4 example.
    assert "6.4" in result.output
    assert "p06_sample_dpa_xfer_001" in result.output


def test_gold_list_all_pillars(runner: CliRunner) -> None:
    result = runner.invoke(app, ["gold", "list", "--repo-root", str(REPO_ROOT)])
    assert result.exit_code == 0, result.output
    # Both shipped pillars surface in the output.
    assert "pillar 6" in result.output
    assert "pillar 7" in result.output
    assert "total:" in result.output


def test_metrics_no_db_path(runner: CliRunner) -> None:
    """`metrics --no-db` short-circuits the DB and reports a zero-claim
    baseline so the CLI works in CI without Postgres."""
    result = runner.invoke(
        app,
        ["metrics", "--pillar", "7", "--no-db", "--repo-root", str(REPO_ROOT)],
    )
    assert result.exit_code == 0, result.output
    assert "count_total" in result.output
    assert "f1_macro" in result.output


def test_recall_command(runner: CliRunner, tmp_path: Path, frozen_repo_root: Path) -> None:
    # Point at a frozen 3-item Pillar-7 gold repo so the assertion is stable as
    # the live Round-1 gold corpus grows.
    payload = tmp_path / "retrieved.json"
    payload.write_text(
        '[["sample_dpa_2020_s6"], ["sample_dpa_2020_s33"], ["sample_dpa_2020_s7"]]',
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        [
            "recall",
            "--pillar",
            "7",
            "--retrieved",
            str(payload),
            "--repo-root",
            str(frozen_repo_root),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "retrieval_recall" in result.output
    assert "1.0000" in result.output


def test_recall_command_rejects_missing_file(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "recall",
            "--pillar",
            "7",
            "--retrieved",
            str(tmp_path / "does_not_exist.json"),
            "--repo-root",
            str(REPO_ROOT),
        ],
    )
    assert result.exit_code == 1
