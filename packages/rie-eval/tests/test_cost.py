"""Unit tests for cost / latency tracking."""

from __future__ import annotations

import time

import pytest
from rie_eval.cost import CostTracker, RunCostReport, summarise_run_cost

# ─── summarise_run_cost ─────────────────────────────────────────────────────


def test_summarise_counts_kinds_and_seconds() -> None:
    events = [
        {"kind": "model_call", "node": "classify", "seconds": 1.5},
        {"kind": "model_call", "node": "verify.entailment", "seconds": 2.5},
        {"kind": "embed", "node": "retrieval", "seconds": 0.5},
        {"kind": "node", "node": "regime_assembly", "seconds": 0.1},
    ]
    report = summarise_run_cost(events, jurisdiction="SAMPLE", pillar_ids=("6", "7"), claim_count=4)
    assert isinstance(report, RunCostReport)
    assert report.jurisdiction == "SAMPLE"
    assert report.pillar_ids == ("6", "7")
    assert report.model_calls == 2
    assert report.model_seconds == pytest.approx(4.0)
    assert report.embedder_calls == 1
    # No explicit total → sum of all event seconds.
    assert report.total_seconds == pytest.approx(4.6)
    assert report.per_claim_seconds == pytest.approx(4.6 / 4)
    # 4 model_seconds → 4/60 minutes.
    assert report.model_minutes_per_jurisdiction == pytest.approx(4.0 / 60.0)


def test_summarise_respects_explicit_total_seconds() -> None:
    events = [{"kind": "model_call", "node": "x", "seconds": 1.0}]
    report = summarise_run_cost(
        events,
        jurisdiction="SAMPLE",
        pillar_ids=("7",),
        claim_count=2,
        total_seconds=10.0,
    )
    assert report.total_seconds == pytest.approx(10.0)
    assert report.per_claim_seconds == pytest.approx(5.0)


def test_summarise_zero_claims_yields_zero_per_claim() -> None:
    report = summarise_run_cost(
        [{"kind": "model_call", "node": "x", "seconds": 3.0}],
        jurisdiction="SAMPLE",
        pillar_ids=(),
        claim_count=0,
    )
    assert report.per_claim_seconds == pytest.approx(0.0)
    assert report.model_minutes_per_jurisdiction == pytest.approx(3.0 / 60.0)


def test_summarise_empty_events_safe() -> None:
    report = summarise_run_cost([], jurisdiction="X", pillar_ids=(), claim_count=0)
    assert report.total_seconds == pytest.approx(0.0)
    assert report.model_calls == 0
    assert report.embedder_calls == 0


# ─── CostTracker ────────────────────────────────────────────────────────────


def test_tracker_records_wall_clock_elapsed() -> None:
    with CostTracker() as t:
        time.sleep(0.05)
    assert t.elapsed_seconds >= 0.05


def test_tracker_helpers_record_correct_kinds() -> None:
    with CostTracker() as t:
        t.model_call("classify", seconds=0.4)
        t.embed("retrieval", seconds=0.01)
        t.node("verify", seconds=0.1)
    kinds = [e["kind"] for e in t.events]
    assert kinds == ["model_call", "embed", "node"]


def test_tracker_summarise_uses_wall_clock_total() -> None:
    with CostTracker() as t:
        t.model_call("classify", seconds=0.4)
        time.sleep(0.05)
    report = t.summarise(jurisdiction="SAMPLE", pillar_ids=("7",), claim_count=2)
    assert report.model_calls == 1
    assert report.model_seconds == pytest.approx(0.4)
    # Wall-clock total is at least the sleep, NOT the sum of event seconds.
    assert report.total_seconds >= 0.05
    # per_claim derived from wall-clock total, not from event sum.
    assert report.per_claim_seconds == pytest.approx(report.total_seconds / 2)


def test_tracker_summarise_computes_model_minutes() -> None:
    with CostTracker() as t:
        t.model_call("a", seconds=30.0)
        t.model_call("b", seconds=30.0)
    report = t.summarise(jurisdiction="SAMPLE", pillar_ids=("7",), claim_count=1)
    # 60 seconds total → 1.0 minute.
    assert report.model_minutes_per_jurisdiction == pytest.approx(1.0)


def test_tracker_pre_enter_elapsed_is_zero() -> None:
    t = CostTracker()
    assert t.elapsed_seconds == pytest.approx(0.0)
