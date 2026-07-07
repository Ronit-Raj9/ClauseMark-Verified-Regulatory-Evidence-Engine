"""Unit tests for tools/cost_logger.py.

The module lives in the repo-root ``tools/`` dir (not an installed package), so
we load it by file path via importlib. Tests are deterministic — fixed events,
fixed ``measured_on`` — and assert correct totals under BOTH stacks plus the
full report schema.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
COST_LOGGER_PATH = REPO_ROOT / "tools" / "cost_logger.py"


def _load_cost_logger() -> ModuleType:
    spec = importlib.util.spec_from_file_location("cost_logger", COST_LOGGER_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cl = _load_cost_logger()


# A representative 50-page benchmark document's measured events.
def _benchmark_events() -> list[dict[str, Any]]:
    return [
        {"component": "crawl", "pages": 12, "seconds": 9.5},
        {"component": "ocr", "engine": "tesseract", "pages": 50, "seconds": 31.2},
        {"component": "embedding", "model": "bge-m3", "tokens": 120000, "seconds": 4.1},
        {
            "component": "llm",
            "model": "qwen3-32b",
            "input_tokens": 80000,
            "output_tokens": 6000,
            "seconds": 22.0,
        },
    ]


# ─── Schema: every required key present ──────────────────────────────────────


def test_report_has_all_schema_keys() -> None:
    report = cl.summarise_events(
        _benchmark_events(), document="benchmark_50pg.pdf", measured_on="2026-06-07"
    )
    for top in (
        "document",
        "measured_on",
        "ocr",
        "embedding",
        "llm",
        "crawl",
        "total_cost_usd",
        "total_cost_usd_open_weight",
        "processing_time_seconds",
    ):
        assert top in report, f"missing top-level key: {top}"
    assert set(report["ocr"]) >= {"engine", "pages", "cost_usd"}
    assert set(report["embedding"]) >= {"model", "tokens", "cost_usd"}
    assert set(report["llm"]) >= {"model", "input_tokens", "output_tokens", "cost_usd"}
    assert set(report["crawl"]) >= {"pages", "cost_usd"}


def test_measured_on_is_passed_through_not_now() -> None:
    report = cl.summarise_events([], document="x", measured_on="2020-01-01")
    assert report["measured_on"] == "2020-01-01"


def test_document_label_recorded() -> None:
    report = cl.summarise_events([], document="my_doc.pdf", measured_on="2026-06-07")
    assert report["document"] == "my_doc.pdf"


# ─── Correct totals (current stack) ──────────────────────────────────────────


def test_current_stack_totals_match_price_table() -> None:
    events = _benchmark_events()
    report = cl.summarise_events(events, document="b", measured_on="2026-06-07")
    cur = cl.PRICE_TABLE["current"]

    exp_ocr = 50 * cur["ocr"]["per_page"]
    exp_emb = (120000 / 1000.0) * cur["embedding"]["per_1k_tokens"]
    exp_llm = (80000 / 1000.0) * cur["llm"]["per_1k_input"] + (6000 / 1000.0) * cur["llm"][
        "per_1k_output"
    ]
    exp_crawl = 12 * cur["crawl"]["per_page"]
    exp_total = exp_ocr + exp_emb + exp_llm + exp_crawl

    assert report["ocr"]["cost_usd"] == pytest.approx(round(exp_ocr, 6))
    assert report["embedding"]["cost_usd"] == pytest.approx(round(exp_emb, 6))
    assert report["llm"]["cost_usd"] == pytest.approx(round(exp_llm, 6))
    assert report["crawl"]["cost_usd"] == pytest.approx(round(exp_crawl, 6))
    assert report["total_cost_usd"] == pytest.approx(round(exp_total, 6))


# ─── Both stacks present; open-weight is zero on the default table ────────────


def test_open_weight_swap_totals_are_zero() -> None:
    report = cl.summarise_events(_benchmark_events(), document="b", measured_on="2026-06-07")
    assert report["total_cost_usd_open_weight"] == 0.0
    assert report["ocr"]["cost_usd_open_weight"] == 0.0
    assert report["embedding"]["cost_usd_open_weight"] == 0.0
    assert report["llm"]["cost_usd_open_weight"] == 0.0
    # And the current stack is strictly more expensive (the cost-efficiency story).
    assert report["total_cost_usd"] > report["total_cost_usd_open_weight"]


# ─── Aggregated quantities ───────────────────────────────────────────────────


def test_quantities_aggregate_across_events() -> None:
    events = [
        {"component": "ocr", "engine": "tesseract", "pages": 20, "seconds": 5.0},
        {"component": "ocr", "engine": "tesseract", "pages": 30, "seconds": 7.0},
        {
            "component": "llm",
            "model": "qwen3-32b",
            "input_tokens": 1000,
            "output_tokens": 100,
            "seconds": 1.0,
        },
        {
            "component": "llm",
            "model": "qwen3-32b",
            "input_tokens": 2000,
            "output_tokens": 200,
            "seconds": 2.0,
        },
    ]
    report = cl.summarise_events(events, document="b", measured_on="2026-06-07")
    assert report["ocr"]["pages"] == 50
    assert report["ocr"]["engine"] == "tesseract"
    assert report["llm"]["input_tokens"] == 3000
    assert report["llm"]["output_tokens"] == 300


def test_processing_time_is_sum_of_event_seconds() -> None:
    report = cl.summarise_events(_benchmark_events(), document="b", measured_on="2026-06-07")
    assert report["processing_time_seconds"] == pytest.approx(9.5 + 31.2 + 4.1 + 22.0)


# ─── Determinism ─────────────────────────────────────────────────────────────


def test_output_is_deterministic() -> None:
    e = _benchmark_events()
    a = cl.summarise_events(e, document="b", measured_on="2026-06-07")
    b = cl.summarise_events(e, document="b", measured_on="2026-06-07")
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_empty_events_yields_zero_costs_valid_schema() -> None:
    report = cl.summarise_events([], document="empty", measured_on="2026-06-07")
    assert report["total_cost_usd"] == 0.0
    assert report["total_cost_usd_open_weight"] == 0.0
    assert report["ocr"]["pages"] == 0
    assert report["processing_time_seconds"] == 0.0


def test_unknown_component_ignored_for_cost_but_counted_in_seconds() -> None:
    events = [{"component": "mystery", "seconds": 3.0}]
    report = cl.summarise_events(events, document="b", measured_on="2026-06-07")
    assert report["total_cost_usd"] == 0.0
    assert report["processing_time_seconds"] == pytest.approx(3.0)


# ─── load_events + file round-trip via main() ────────────────────────────────


def test_load_events_accepts_list_and_wrapped(tmp_path: Path) -> None:
    list_path = tmp_path / "list.json"
    list_path.write_text(json.dumps(_benchmark_events()), encoding="utf-8")
    assert len(cl.load_events(list_path)) == 4

    wrapped_path = tmp_path / "wrapped.json"
    wrapped_path.write_text(json.dumps({"events": _benchmark_events()}), encoding="utf-8")
    assert len(cl.load_events(wrapped_path)) == 4


def test_main_writes_report_file(tmp_path: Path) -> None:
    events_path = tmp_path / "events.json"
    events_path.write_text(json.dumps(_benchmark_events()), encoding="utf-8")
    out_path = tmp_path / "logs" / "cost_report.json"
    rc = cl.main(
        [
            "--events",
            str(events_path),
            "--measured-on",
            "2026-06-07",
            "--document",
            "benchmark_50pg.pdf",
            "--out",
            str(out_path),
        ]
    )
    assert rc == 0
    assert out_path.exists()
    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written["document"] == "benchmark_50pg.pdf"
    assert written["measured_on"] == "2026-06-07"
    assert written["total_cost_usd"] > 0.0


def test_main_missing_events_file_returns_one(tmp_path: Path) -> None:
    rc = cl.main(["--events", str(tmp_path / "nope.json"), "--measured-on", "2026-06-07"])
    assert rc == 1


def test_main_pdf_without_events_emits_empty_report(tmp_path: Path) -> None:
    out_path = tmp_path / "cost_report.json"
    rc = cl.main(
        [
            "--pdf",
            "some.pdf",
            "--economy",
            "Singapore",
            "--pillar",
            "6",
            "--measured-on",
            "2026-06-07",
            "--out",
            str(out_path),
        ]
    )
    assert rc == 0
    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written["document"] == "some.pdf"
    assert written["total_cost_usd"] == 0.0
