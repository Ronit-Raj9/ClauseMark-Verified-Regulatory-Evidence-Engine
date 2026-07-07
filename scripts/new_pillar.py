#!/usr/bin/env python3
"""Scaffold a new pillar from the template.

Usage:
    uv run python scripts/new_pillar.py \
        --id 08 \
        --name "Internet intermediary liability" \
        --cluster digital_governance \
        --profile statutory_legal_text

Three files touched: pillar YAML, registry.yaml, gold/<dir>/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "pillars" / "_template" / "pillar_NN_template.yaml"
REGISTRY = ROOT / "pillars" / "registry.yaml"
CLUSTERS = {"digital_governance", "traditional_trade", "other_domestic"}
PROFILES = {
    "statutory_legal_text",
    "structured_tabular",
    "mixed_regulatory",
    "treaty_membership",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scaffold a new pillar.")
    p.add_argument("--id", required=True, help="Pillar id, e.g. 08")
    p.add_argument("--name", required=True)
    p.add_argument("--cluster", required=True, choices=sorted(CLUSTERS))
    p.add_argument("--profile", required=True, choices=sorted(PROFILES))
    return p.parse_args()


def slugify(name: str) -> str:
    out = []
    for ch in name.lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in {" ", "-", "_"}:
            out.append("_")
    s = "".join(out)
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def main() -> int:
    args = parse_args()
    pid_int = int(args.id)
    pid_str = str(pid_int)
    slug = slugify(args.name)
    out_dir = ROOT / "pillars" / args.cluster
    out_yaml = out_dir / f"pillar_{pid_int:02d}_{slug}.yaml"
    gold_dir = ROOT / "gold" / f"pillar_{pid_int:02d}"

    if out_yaml.exists():
        print(f"REFUSE: {out_yaml} already exists")
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    gold_dir.mkdir(parents=True, exist_ok=True)

    text = TEMPLATE.read_text(encoding="utf-8")
    text = text.replace('"NN"', f'"{pid_str}"')
    text = text.replace('"NN.1"', f'"{pid_str}.1"')
    text = text.replace('"Replace me"', f'"{args.name}"', 1)
    text = text.replace('cluster: "digital_governance"', f'cluster: "{args.cluster}"')
    text = text.replace(
        'document_profile: "statutory_legal_text"',
        f'document_profile: "{args.profile}"',
    )
    out_yaml.write_text(text, encoding="utf-8")

    registry = yaml.safe_load(REGISTRY.read_text())
    if any(p["pillar_id"] == pid_str for p in registry["pillars"]):
        print(f"REFUSE: pillar_id {pid_str} already in registry")
        return 1
    registry["pillars"].append(
        {
            "pillar_id": pid_str,
            "pillar_name": args.name,
            "cluster": args.cluster,
            "document_profile": args.profile,
            "status": "stub",
            "config_path": str(out_yaml.relative_to(ROOT)),
            "gold_set_path": str(gold_dir.relative_to(ROOT)),
        }
    )
    REGISTRY.write_text(
        yaml.safe_dump(registry, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    print(f"✓ wrote pillar config: {out_yaml.relative_to(ROOT)}")
    print(f"✓ created gold dir:    {gold_dir.relative_to(ROOT)}")
    print(f"✓ registered:          pillar_id={pid_str}")
    print("\nNext: populate the YAML, drop a gold set, flip status: built.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
