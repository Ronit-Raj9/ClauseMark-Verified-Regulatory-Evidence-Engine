# Adding an RDTII pillar

This is the canonical walkthrough. Implementation plan §6 promises that adding a pillar
touches **three files** (with one strategy file if a new document profile is needed).

## TL;DR

```bash
# Scaffold and register
uv run python scripts/new_pillar.py --id 08 --name "Internet intermediary liability" \
  --cluster digital_governance --profile statutory_legal_text
```

This will:
1. Copy `pillars/_template/pillar_NN_template.yaml` to
   `pillars/<cluster>/pillar_08_internet_intermediary_liability.yaml`.
2. Add an entry to `pillars/registry.yaml`.
3. Create `gold/pillar_08/` if missing.

Then a human:
1. Populates indicator definitions / scoring criteria / few-shot examples in the
   new YAML.
2. Drops gold items into `gold/pillar_08/<jurisdiction>.yaml`.
3. Flips `status: stub` → `status: built` in `registry.yaml` once the gold set is
   non-empty.

That is the entire change to extend the engine to the new pillar.

## New document profile?

If the new pillar declares a `document_profile` not yet in
`packages/rie-profiles/src/rie_profiles/strategies/`, the Substantive/Technical Lead
implements ONE new strategy file (subclass-by-protocol of `DocumentProfileStrategy`)
and registers it in `strategies/__init__.py`. Nothing else in the engine moves —
retrieval, classification, verification, persistence and UI all stay untouched.

## What "built" buys you

Once `status: built`:
- The orchestrator iterates the registry; the new pillar is picked up automatically.
- `make demo` and `python scripts/run_jurisdiction.py --pillars <id>` work.
- `rie-eval metrics --pillar <id>` reports per-indicator PRF + the §8 metrics.

## What does NOT happen automatically

- Authority overrides are jurisdiction-specific; populate
  `authority_overrides` rows in Postgres per `data/seeds/authority_overrides.yaml`.
- Source registry (`sources/jurisdictions/<jurisdiction>.yaml`) must list the
  documents the engine will see.

## Hard rules

- The engine code does NOT reference your pillar id. If you find yourself editing
  Python to "support" a pillar, you are doing it wrong — fix the pillar YAML or
  the profile strategy instead.
- Pillar YAMLs validate against `pillars/_schema/pillar.schema.json`. CI fails on
  schema violation.
- The gold set is the legal-correctness contract. A built pillar with an empty
  gold set is a bug.
