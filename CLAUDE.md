# Repo-wide rules for Claude Code agents

Every agent invoked in this repo MUST obey the following. These are not suggestions.

## Model & runtime
- **Model:** every agent runs `claude-opus-4-7` at **high reasoning effort**.
- **Python 3.12** only. `requires-python = ">=3.12,<3.13"`.
- **uv only.** Use `uv add`, `uv sync`, `uv run`. Never `pip install`. Never hand-edit `uv.lock`.

## File ownership
- **Stay in your package.** Edit only files under your assigned `packages/<name>/`.
- Data agents (`config`, `eval`) may also edit `pillars/`, `sources/`, `gold/`.
- `test-agent` owns `tests/` (contract + integration + architecture).
- **`rie-domain` and `rie-orchestration` are lead-owned** — agents never edit them.

## Contracts
- `rie-contracts` is **FROZEN** after Phase 0. Never modify it. If a contract seems wrong,
  STOP and escalate; do not work around it.
- Every adapter MUST pass its contract test (`tests/contract/test_<port>.py`) before its PR.

## Dependency direction (CI-enforced)
- `rie-contracts` imports nothing else in the workspace.
- `rie-domain` imports only `rie-contracts`.
- Adapters (`rie-extract`, `rie-retrieval`, …) import `rie-contracts` and `rie-profiles`.
- `rie-orchestration` imports any adapter through its port; never reaches inside.
- `rie-api` and `rie-ui` import `rie-orchestration` and `rie-contracts` only.
- A violation FAILS the build via `tests/architecture/`.

## Two-layer output (anti-hallucination keystone)
- **The LLM never authors a fact.** Not a citation, not an authority tier, not a score.
- It selects from enums (constrained decoding) and emits **span IDs** (e.g. `doc12#4120-4215`).
- Deterministic code does ID-replacement to materialize citations from stored metadata.
- Layer 1 (extraction): verifiable, automatable — never emitted without all 4 gates passing.
- Layer 2 (score): a **recommendation** with `human_confirmation_required = True`.

## Pillar discipline
- **No pillar logic in `.py` files.** No hard-coded `indicator_id == "6.4"`, no Python
  `if "cross-border" in name`. Behaviour lives in `pillars/*.yaml`.
- A hard-coded pillar id in engine code is a **bug** and CI catches it.

## Typing & style
- `pydantic >= 2` for every data model. `Protocol` for every port.
- `pyright` strict mode. `ruff check` + `ruff format` clean before commit.
- No `Any` in public signatures. No untyped `dict[str, ...]` returned from a port.

## Verification gates (the 4)
1. **Span-existence** — `char_offsets` resolve to real text. Deterministic.
2. **Verbatim-match** — exported snippet byte-identical to re-extracted span. Deterministic.
3. **Entailment** — NLI + second LLM; **any disagreement → `flagged`**, never auto-resolved.
4. **Self-consistency** — N=3 sampling on FIXED input; unstable label → `flagged`.

KG / entity-grounding is **Phase 2 roadmap**, not MVP.

## Absence reasoning
- Never emit a bare `0`. Three states only: `evidence_found`, `no_evidence_in_searched_corpus`
  (carries measured gold-set recall), `insufficient_coverage`.

## Tests
- Every adapter has a `tests/contract/test_<port>.py` running against the port — passes with
  any conforming implementation.
- `tests/integration/` runs the full graph on `tests/fixtures/`.
- `tests/architecture/test_dependency_directions.py` is a fitness function; do not weaken it.

## Commits
- One package per PR. Disjoint file sets. No cross-package edits in a single commit.
- Commit message: conventional commits, e.g. `feat(extract): add Docling adapter with structure graph`.

## Forbidden
- `pip install <anything>` — use `uv add` in the relevant package.
- Importing concrete adapter classes from another package — go through ports.
- Mutating `rie-contracts` after freeze.
- Hard-coding a pillar id, indicator id, or jurisdiction in engine code.
- LLM-authored citations or scores.
- Silent `0` scores.
