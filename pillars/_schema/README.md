# Pillar config schema

Every file under `pillars/<cluster>/pillar_NN_*.yaml` validates against:

- `pillar.schema.json` — root shape
- `indicator.schema.json` — per-indicator shape

CI fails any malformed pillar via `rie-config` schema validation +
`tests/integration/test_pillar_validation.py`.

## Authoring a new pillar

1. Copy `pillars/_template/pillar_NN_template.yaml`.
2. Fill in the indicator definitions, scoring criteria, keywords, few-shot examples.
3. Register the pillar in `pillars/registry.yaml`.
4. Drop a gold set folder at `gold/pillar_NN/`.
5. Flip `status: stub` → `status: built` once gold set is non-empty.

See `docs/guides/adding_a_pillar.md` for the full walkthrough.
