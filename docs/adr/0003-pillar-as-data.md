# ADR-0003: Pillar behaviour is data, never code

**Status:** Accepted

## Context

Twelve RDTII pillars exist; the engine must scale across all of them without
becoming a if/else explosion.

## Decision

A pillar is a YAML file, an entry in `pillars/registry.yaml`, a gold set, and
(if its document profile is new) one strategy class. **No Python file in
`packages/` references a specific pillar id, indicator id, or jurisdiction.**

`tests/architecture/test_no_hardcoded_indicator_ids` enforces this.

## Consequences

- Adding a pillar touches THREE files.
- "Run only built pillars" is a one-line registry query.
- A hard-coded indicator id in engine code FAILS CI.
