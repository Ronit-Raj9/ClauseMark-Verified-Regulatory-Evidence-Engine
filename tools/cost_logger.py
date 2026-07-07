#!/usr/bin/env python3
"""tools/cost_logger.py — measured per-document cost report.

Architecture-track deliverable: judges verify *actual measured* operational
cost per document against code (not an estimate). This wraps a run and records
per-component measured cost + tokens + seconds for the four cost centres:

    ocr  ·  embedding  ·  llm  ·  crawl

It then renders ``logs/cost_report.json`` in the judge-facing schema and prints
a short summary. Costs are computed from a small **editable** ``PRICE_TABLE``
so swapping the stack (current managed APIs ↔ open-weight self-hosted) is a
data change, not a code change — the report shows BOTH totals so the
cost-efficiency story is explicit.

Determinism
-----------
``measured_on`` is taken from ``--measured-on`` (an ISO date string the caller
supplies); the tool NEVER calls ``datetime.now()`` / ``Date.now()`` and uses no
randomness. Given the same events file the output bytes are identical, so the
report is unit-testable.

Two input modes
---------------
1. ``--events events.json``  (deterministic, testable, no live deps)
   A JSON list of measured-event records. Each record has a ``component`` key
   in ``{ocr, embedding, llm, crawl}`` plus the measured fields for that
   component (see EVENT SCHEMA below). This is the mode the test-suite exercises
   and the mode to use when re-summarising captured telemetry.

2. ``--pdf X --economy Y --pillar 6``  (live wrapper)
   Placeholder for an in-process run. The live graph is owned by other
   packages; rather than reach across a port, this script accepts the run's
   measured events through ``--events`` produced by that run. When invoked with
   ``--pdf`` and no ``--events`` it emits an empty-but-valid report (all zeros)
   so the CLI contract still holds and a real integration can drop measured
   events in later. If ``rie_eval.cost.CostTracker`` is importable it is used to
   sum wall-clock seconds across event seconds.

EVENT SCHEMA (one dict per measured component invocation)
---------------------------------------------------------
    {"component": "ocr",       "engine": "tesseract", "pages": 50, "seconds": 31.2}
    {"component": "embedding", "model": "bge-m3",     "tokens": 120000, "seconds": 4.1}
    {"component": "llm",       "model": "qwen3-32b",  "input_tokens": 80000,
                                                       "output_tokens": 6000, "seconds": 22.0}
    {"component": "crawl",     "pages": 12,            "seconds": 9.5}

Unknown/extra keys are ignored; missing numeric fields default to 0.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

# ─── PRICE TABLE (editable) ──────────────────────────────────────────────────
# Per-unit USD rates. Two stacks side-by-side. Open-weight = self-hosted
# open models / engines; the marginal API cost is 0.0 (compute is amortised
# elsewhere and not a per-document API charge). Edit these freely — they are
# data, not logic.
#
# Units:
#   ocr:        USD per page
#   embedding:  USD per 1K tokens
#   llm:        USD per 1K input tokens / per 1K output tokens
#   crawl:      USD per page fetched
PRICE_TABLE: dict[str, dict[str, dict[str, float]]] = {
    "current": {
        # Managed / commercial stack (illustrative public list prices).
        "ocr": {"per_page": 0.0015},  # e.g. cloud OCR ~ $1.50 / 1000 pages
        "embedding": {"per_1k_tokens": 0.00002},  # e.g. small embedding model
        "llm": {"per_1k_input": 0.00015, "per_1k_output": 0.0006},  # mid-tier API
        "crawl": {"per_page": 0.0},  # HTTP fetch — no per-page API charge
    },
    "open_weight": {
        # Open-weight self-hosted swap: tesseract OCR, bge embeddings, a
        # self-hosted Llama/Qwen. Marginal per-document API cost = 0.0.
        "ocr": {"per_page": 0.0},
        "embedding": {"per_1k_tokens": 0.0},
        "llm": {"per_1k_input": 0.0, "per_1k_output": 0.0},
        "crawl": {"per_page": 0.0},
    },
}

VALID_COMPONENTS = ("ocr", "embedding", "llm", "crawl")


# ─── Number coercion ─────────────────────────────────────────────────────────


def _num(rec: Mapping[str, Any], key: str) -> float:
    v = rec.get(key, 0)
    try:
        return float(v) if v is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _int(rec: Mapping[str, Any], key: str) -> int:
    return int(_num(rec, key))


def _str(rec: Mapping[str, Any], key: str, default: str = "") -> str:
    v = rec.get(key)
    return str(v) if v is not None else default


# ─── Per-component cost ──────────────────────────────────────────────────────


def _ocr_cost(pages: int, prices: Mapping[str, float]) -> float:
    return pages * prices.get("per_page", 0.0)


def _embedding_cost(tokens: int, prices: Mapping[str, float]) -> float:
    return (tokens / 1000.0) * prices.get("per_1k_tokens", 0.0)


def _llm_cost(input_tokens: int, output_tokens: int, prices: Mapping[str, float]) -> float:
    return (input_tokens / 1000.0) * prices.get("per_1k_input", 0.0) + (
        output_tokens / 1000.0
    ) * prices.get("per_1k_output", 0.0)


def _crawl_cost(pages: int, prices: Mapping[str, float]) -> float:
    return pages * prices.get("per_page", 0.0)


# ─── Aggregation ─────────────────────────────────────────────────────────────


def summarise_events(
    events: Sequence[Mapping[str, Any]],
    *,
    document: str,
    measured_on: str,
    price_table: Mapping[str, Any] = PRICE_TABLE,
) -> dict[str, Any]:
    """Aggregate measured events into the judge-facing cost-report dict.

    Pure + deterministic: same events + same ``measured_on`` ⇒ same output.
    Computes per-component cost under BOTH the ``current`` and ``open_weight``
    stacks. The top-level ``total_cost_usd`` reflects the **current** stack
    (the live, billed cost); ``total_cost_usd_open_weight`` shows the swap.
    """
    # Aggregate measured quantities across all events of each component.
    ocr_pages = 0
    ocr_engine = ""
    emb_tokens = 0
    emb_model = ""
    llm_in = 0
    llm_out = 0
    llm_model = ""
    crawl_pages = 0
    total_seconds = 0.0

    for rec in events:
        comp = _str(rec, "component").strip().lower()
        total_seconds += _num(rec, "seconds")
        if comp == "ocr":
            ocr_pages += _int(rec, "pages")
            ocr_engine = ocr_engine or _str(rec, "engine")
        elif comp == "embedding":
            emb_tokens += _int(rec, "tokens")
            emb_model = emb_model or _str(rec, "model")
        elif comp == "llm":
            llm_in += _int(rec, "input_tokens")
            llm_out += _int(rec, "output_tokens")
            llm_model = llm_model or _str(rec, "model")
        elif comp == "crawl":
            crawl_pages += _int(rec, "pages")
        # Unknown component → ignored for cost, still counted in seconds.

    cur = price_table["current"]
    opn = price_table["open_weight"]

    ocr_cost_cur = _ocr_cost(ocr_pages, cur["ocr"])
    emb_cost_cur = _embedding_cost(emb_tokens, cur["embedding"])
    llm_cost_cur = _llm_cost(llm_in, llm_out, cur["llm"])
    crawl_cost_cur = _crawl_cost(crawl_pages, cur["crawl"])
    total_cur = ocr_cost_cur + emb_cost_cur + llm_cost_cur + crawl_cost_cur

    ocr_cost_opn = _ocr_cost(ocr_pages, opn["ocr"])
    emb_cost_opn = _embedding_cost(emb_tokens, opn["embedding"])
    llm_cost_opn = _llm_cost(llm_in, llm_out, opn["llm"])
    crawl_cost_opn = _crawl_cost(crawl_pages, opn["crawl"])
    total_opn = ocr_cost_opn + emb_cost_opn + llm_cost_opn + crawl_cost_opn

    return {
        "document": document,
        "measured_on": measured_on,
        "ocr": {
            "engine": ocr_engine,
            "pages": ocr_pages,
            "cost_usd": round(ocr_cost_cur, 6),
            "cost_usd_open_weight": round(ocr_cost_opn, 6),
        },
        "embedding": {
            "model": emb_model,
            "tokens": emb_tokens,
            "cost_usd": round(emb_cost_cur, 6),
            "cost_usd_open_weight": round(emb_cost_opn, 6),
        },
        "llm": {
            "model": llm_model,
            "input_tokens": llm_in,
            "output_tokens": llm_out,
            "cost_usd": round(llm_cost_cur, 6),
            "cost_usd_open_weight": round(llm_cost_opn, 6),
        },
        "crawl": {
            "pages": crawl_pages,
            "cost_usd": round(crawl_cost_cur, 6),
            "cost_usd_open_weight": round(crawl_cost_opn, 6),
        },
        "total_cost_usd": round(total_cur, 6),
        "total_cost_usd_open_weight": round(total_opn, 6),
        "processing_time_seconds": round(_wall_seconds(events, total_seconds), 4),
    }


def _wall_seconds(events: Sequence[Mapping[str, Any]], summed_seconds: float) -> float:
    """Total processing seconds.

    If ``rie_eval.cost.CostTracker`` is importable we feed the per-event seconds
    through ``summarise_run_cost`` (so the same summing logic the orchestrator
    uses is reused); otherwise fall back to the plain sum. Either way the value
    is the additive sum of measured component seconds — deterministic.
    """
    try:
        # Deferred + optional: reuse the orchestrator's summariser when present,
        # but keep this script runnable with zero workspace deps.
        from rie_eval.cost import summarise_run_cost  # noqa: PLC0415
    except Exception:
        return summed_seconds
    # Map our events onto the cost-tracker event shape (kind/seconds) so the
    # shared summariser does the addition. Component → "node" kind is fine; we
    # only consume total_seconds here.
    mapped = [
        {"kind": "node", "node": _str(e, "component"), "seconds": _num(e, "seconds")}
        for e in events
    ]
    report = summarise_run_cost(mapped)
    return report.total_seconds


# ─── I/O ─────────────────────────────────────────────────────────────────────


def load_events(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("events") or payload.get("items") or []
    if not isinstance(payload, list) or any(not isinstance(r, dict) for r in payload):
        raise ValueError("events file must be a JSON list of record objects")
    return payload


def write_report(report: Mapping[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# ─── CLI ─────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cost_logger.py",
        description="Measured per-document cost report (current vs open-weight stack).",
    )
    p.add_argument("--pdf", help="Source document path/identifier (live-run label).")
    p.add_argument("--economy", help="Economy name (live-run context).")
    p.add_argument("--pillar", help="Pillar id (live-run context).")
    p.add_argument(
        "--events",
        type=Path,
        help="JSON file of measured events (deterministic mode). See module docstring.",
    )
    p.add_argument(
        "--measured-on",
        required=True,
        help="ISO date the measurement was taken (e.g. 2026-06-07). Passed in, NOT now().",
    )
    p.add_argument(
        "--document",
        help="Document label for the report. Defaults to --pdf, else 'unknown'.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("logs/cost_report.json"),
        help="Output path (default: logs/cost_report.json).",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    document = args.document or args.pdf or "unknown"

    events: list[dict[str, Any]]
    if args.events is not None:
        if not args.events.exists():
            print(f"missing events file: {args.events}", file=sys.stderr)
            return 1
        try:
            events = load_events(args.events)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
    else:
        # Live-run wrapper with no captured events yet → emit a valid empty
        # report so the CLI contract holds and a real run can supply events.
        print(
            "no --events supplied; emitting empty-but-valid report. "
            "Provide measured events via --events for real numbers.",
            file=sys.stderr,
        )
        events = []

    report = summarise_events(events, document=document, measured_on=args.measured_on)
    write_report(report, args.out)

    # Short human summary to stdout.
    print(f"cost_report → {args.out}")
    print(
        f"  document={report['document']}  measured_on={report['measured_on']}  "
        f"time={report['processing_time_seconds']}s"
    )
    print(f"  total_cost_usd (current)     = {report['total_cost_usd']}")
    print(f"  total_cost_usd (open-weight) = {report['total_cost_usd_open_weight']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
