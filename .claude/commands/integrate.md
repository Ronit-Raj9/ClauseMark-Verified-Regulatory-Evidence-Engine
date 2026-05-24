---
name: integrate
description: End-to-end run + metrics. Phase 4.
---

Steps:
1. `make up` — start Postgres + Qdrant.
2. `make migrate`.
3. `make demo` — runs `scripts/run_jurisdiction.py --jurisdiction SAMPLE --pillars 6,7`.
4. `uv run pytest tests/integration -q`.
5. `uv run python -m rie_eval metrics --pillar 6` and `--pillar 7`.
