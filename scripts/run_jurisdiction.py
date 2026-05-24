#!/usr/bin/env python3
"""Run the full RIE pipeline on a jurisdiction. CLI entrypoint.

Usage:
    uv run python scripts/run_jurisdiction.py --jurisdiction SAMPLE --pillars 6,7

Goes through ingest → extract → retrieval → classify → verify → coverage and
writes the evidence package to data/outputs/<jurisdiction>_<timestamp>.json.

Requires `rie-orchestration` to be available.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the RIE pipeline for a jurisdiction.")
    p.add_argument("--jurisdiction", default="SAMPLE")
    p.add_argument(
        "--pillars",
        default="6,7",
        help="Comma-separated pillar ids to run. Default: 6,7 (MVP).",
    )
    p.add_argument("--output-dir", default=str(ROOT / "data" / "outputs"))
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Wire the graph but skip persistence + LLM calls. Smoke test.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    pillar_ids = [s.strip() for s in args.pillars.split(",") if s.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from rie_orchestration.runner import run_pipeline  # type: ignore[import-not-found]
    except ImportError as e:
        print(f"ERROR: rie_orchestration.runner not available yet: {e}", file=sys.stderr)
        print("       Build Phase 3 first.", file=sys.stderr)
        return 2

    package = run_pipeline(
        jurisdiction=args.jurisdiction,
        pillar_ids=pillar_ids,
        dry_run=args.dry_run,
    )

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = output_dir / f"{args.jurisdiction.lower()}_{stamp}.json"
    out.write_text(json.dumps(package, indent=2, default=str), encoding="utf-8")
    print(f"✓ wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
