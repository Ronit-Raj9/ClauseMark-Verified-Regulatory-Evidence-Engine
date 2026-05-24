---
name: add-pillar
description: Scaffold a new pillar (3-file change).
---

Argument: `$1` = pillar id (e.g. `08`).

Steps:
1. `uv run python scripts/new_pillar.py --id $1 ...` (prompts for name + cluster + profile).
2. Open the new pillar YAML for authoring.
3. Remind the author to drop a gold set and flip `status: built`.
