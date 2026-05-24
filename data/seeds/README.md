# Data seeds

Files in this directory are **governed artefacts** — versioned + signed off by
the Substantive Lead. They populate database tables that the engine consults
deterministically.

## Files

| File | Loads into | Owner |
|---|---|---|
| `authority_overrides.yaml` | `authority_overrides` table | Substantive Lead |

## Applying seeds

```bash
uv run python scripts/seed_authority_overrides.py            # uses DATABASE_URL_SYNC
uv run python scripts/seed_authority_overrides.py --dry-run  # parse + validate only
```

The loader is **upsert**: existing `(jurisdiction, source_pattern)` rows are
updated to match the YAML; rows present in DB but not in YAML are left alone
(seed is additive — a manual `DELETE` removes obsolete rows).
