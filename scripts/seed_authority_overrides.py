#!/usr/bin/env python3
"""Apply data/seeds/authority_overrides.yaml to Postgres.

Idempotent upsert. Dry-run mode parses + validates only (no DB).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VALID_TIERS = {
    "tier_1_statute",
    "tier_2_regulation",
    "tier_3_guideline",
    "tier_4_informal",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Seed authority_overrides table.")
    p.add_argument("--seed", default=str(ROOT / "data" / "seeds" / "authority_overrides.yaml"))
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _validate(rows: list[dict]) -> None:
    seen: set[tuple[str, str]] = set()
    for i, row in enumerate(rows):
        for key in ("jurisdiction", "source_pattern", "authority_tier", "rationale"):
            if key not in row:
                raise SystemExit(f"row {i}: missing {key}")
        if row["authority_tier"] not in VALID_TIERS:
            raise SystemExit(
                f"row {i}: bad tier {row['authority_tier']} (allowed: {sorted(VALID_TIERS)})"
            )
        key = (row["jurisdiction"], row["source_pattern"])
        if key in seen:
            raise SystemExit(f"row {i}: duplicate (jurisdiction, source_pattern) {key}")
        seen.add(key)


def main() -> int:
    args = parse_args()
    seed = Path(args.seed)
    if not seed.exists():
        print(f"ERROR: {seed} missing", file=sys.stderr)
        return 1
    data = yaml.safe_load(seed.read_text())
    if not isinstance(data, dict) or "overrides" not in data:
        print(f"ERROR: {seed} missing 'overrides' key", file=sys.stderr)
        return 1
    rows = data["overrides"]
    _validate(rows)
    print(f"✓ {len(rows)} row(s) validated")

    if args.dry_run:
        return 0

    from rie_persistence.db import create_engine
    from rie_persistence.models import AuthorityOverrideRow
    from sqlalchemy import select
    from sqlalchemy.orm import sessionmaker

    engine = create_engine()
    Session = sessionmaker(engine, future=True, expire_on_commit=False)
    inserted = 0
    updated = 0
    with Session() as s:
        for row in rows:
            existing = s.scalar(
                select(AuthorityOverrideRow).where(
                    AuthorityOverrideRow.jurisdiction == row["jurisdiction"],
                    AuthorityOverrideRow.source_pattern == row["source_pattern"],
                )
            )
            if existing:
                existing.authority_tier = row["authority_tier"]
                existing.rationale = row["rationale"]
                updated += 1
            else:
                s.add(AuthorityOverrideRow(**row))
                inserted += 1
        s.commit()
    print(f"✓ inserted={inserted} updated={updated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
