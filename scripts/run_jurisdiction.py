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


def _load_dotenv(path: Path) -> None:
    """Tiny .env loader. Avoids extra dep, idempotent."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)


_load_dotenv(ROOT / ".env")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the RIE pipeline for a jurisdiction.")
    p.add_argument("--jurisdiction", default="SAMPLE")
    p.add_argument(
        "--pillars",
        default="6,7",
        help="Comma-separated pillar ids to run. Default: 6,7 (MVP). Use 6,7,8,9,12 for digital-governance Phase 2.",
    )
    p.add_argument("--output-dir", default=str(ROOT / "data" / "outputs"))
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Wire the graph but skip persistence + LLM calls. Smoke test.",
    )
    p.add_argument(
        "--kg-gate",
        action="store_true",
        help="Enable Phase 2 KG grounding gate (sets RIE_KG_GATE_ENABLED=1).",
    )
    p.add_argument(
        "--corpus-completeness",
        action="store_true",
        help="Enable Phase 2 corpus completeness downgrade (sets RIE_CORPUS_COMPLETENESS_ENABLED=1).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.kg_gate:
        os.environ["RIE_KG_GATE_ENABLED"] = "1"
    if args.corpus_completeness:
        os.environ["RIE_CORPUS_COMPLETENESS_ENABLED"] = "1"
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
