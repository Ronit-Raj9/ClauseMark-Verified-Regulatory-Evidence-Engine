# ADR-0001: Frozen contracts before parallelism

**Status:** Accepted

## Context

The build mixes work owned by a lead session with work owned by N parallel agents.
Multi-agent builds collapse when agents edit the same files.

## Decision

`packages/rie-contracts` is a **frozen** contract layer of ports (`Protocol`) and
Pydantic models. It is written and approved in Phase 0; no further edits are
permitted after the freeze tag.

Every adapter package implements one of these ports. Agents work behind the
contracts blind to each other because the only shared surface has stopped moving.

## Consequences

- Adding a port requires a deliberate contract freeze bump.
- Architecture fitness function (`tests/architecture/`) enforces dependency
  direction.
- Cross-package edits in a single PR are forbidden.
