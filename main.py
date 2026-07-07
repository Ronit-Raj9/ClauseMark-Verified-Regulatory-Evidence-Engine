#!/usr/bin/env python3
"""RIE CLI — run the real pipeline for one economy + pillar(s).

    python main.py --economy Malaysia --pillar 6
    python main.py --economy Malaysia --pillar 6,7 --local-only
    python main.py --economy Singapore --pillar 6 --out-dir outputs/

Uses the REAL adapter bundle (PyMuPDF/Docling extract, BGE-M3 + Qdrant
retrieval, Ollama classification, NLI + 2nd-LLM verification, Postgres
persistence). Requires the infra in `.env` (Postgres, Qdrant, Ollama). Writes
the judge-validated 13-column CSV + JSON to the output directory.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the RIE pipeline for an economy.")
    p.add_argument("--economy", required=True, help="UN economy name, e.g. Malaysia")
    p.add_argument("--pillar", default="6,7", help="Comma pillar ids, e.g. 6 or 6,7")
    p.add_argument("--out-dir", default=str(ROOT / "outputs"))
    p.add_argument(
        "--local-only",
        action="store_true",
        help="Ingest only registry sources with a local file (offline, no crawl).",
    )
    p.add_argument(
        "--use-fakes",
        action="store_true",
        help="Deterministic in-memory adapters (no LLM/Qdrant/Postgres).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    pillar_ids = [s.strip() for s in args.pillar.split(",") if s.strip()]
    out_dir = Path(args.out_dir)

    try:
        from rie_orchestration import run_pipeline
    except ImportError as e:
        print(f"ERROR: rie_orchestration unavailable: {e}", file=sys.stderr)
        return 2

    package = run_pipeline(
        jurisdiction=args.economy,
        pillar_ids=pillar_ids,
        local_only=args.local_only,
        use_fakes=args.use_fakes,
        enable_hitl=False,  # batch run → no interrupt; FLAGGED still withheld
        enable_postgres_checkpointer=False,
        output_dir=out_dir,
    )

    counts = package.get("counts", {})
    sub = package.get("submission", {})
    print(f"\n economy={args.economy} pillars={pillar_ids}")
    print(f" documents={counts.get('documents')} claims={counts.get('claims')} "
          f"verified={counts.get('verified')} flagged={counts.get('flagged')} "
          f"rejected={counts.get('rejected')}")
    if sub:
        print(f" CSV : {sub['csv']}")
        print(f" JSON: {sub['json']}")
        print(f" rows: {sub['rows']}  schema_violations: {len(sub['violations'])}")
        if sub["violations"]:
            for v in sub["violations"][:10]:
                print(f"   ! {v}")
    print(json.dumps(counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
