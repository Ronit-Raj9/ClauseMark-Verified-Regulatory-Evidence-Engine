"""LangfuseTracer no-op + span recording tests."""

from __future__ import annotations

from rie_orchestration.tracing import LangfuseTracer


def test_tracer_disabled_when_no_keys(monkeypatch) -> None:
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    t = LangfuseTracer.from_env(run_id="r1", jurisdiction="SAMPLE")
    assert t.enabled is False
    t.start()
    with t.span("ingest"):
        pass
    t.stop()
    s = t.summary()
    assert s["enabled"] is False
    assert any(e["name"] == "ingest" for e in s["events"])
    assert s["total_seconds"] >= 0


def test_tracer_records_multiple_spans(monkeypatch) -> None:
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    t = LangfuseTracer.from_env(run_id="r2")
    t.start(metadata={"pillar_ids": ["6"]})
    for name in ("ingest", "extract", "classify"):
        with t.span(name):
            pass
    t.stop()
    s = t.summary()
    assert {e["name"] for e in s["events"]} == {"ingest", "extract", "classify"}


def test_tracer_records_exception(monkeypatch) -> None:
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    t = LangfuseTracer.from_env(run_id="r3")
    t.start()
    try:
        with t.span("verify"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    t.stop()
    s = t.summary()
    evt = next(e for e in s["events"] if e["name"] == "verify")
    assert evt.get("exc") is not None
