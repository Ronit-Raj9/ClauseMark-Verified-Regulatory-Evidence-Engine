# ADR-0002: Two-layer output (extraction vs scoring)

**Status:** Accepted

## Context

Citation hallucination in legal AI runs 17–33% even with citation constraints.
A monolithic "model emits a score" architecture cannot be made safe — there is
no inspectable failure mode between input and output.

## Decision

Two output layers with one honest boundary:
- **Layer 1 (extraction)** — verifiable, automatable, mechanically checked by
  the 4 gates (§6.5). The LLM picks an `indicator_id` from an enum and emits
  span IDs only.
- **Layer 2 (score)** — a *recommendation* carrying `human_confirmation_required = True`.
  The system never emits a final score; the reviewer always does.

## Consequences

- `Layer2Recommendation.human_confirmation_required` is `Literal[True]` in
  contracts — cannot be turned off in code.
- The audit UI's queue prioritises `flagged` claims and `no_evidence` coverage.
- Public claim: *"every Layer-1 citation is mechanically verifiable, and every
  unverifiable claim is withheld from the reviewer's 'verified' queue"* — not
  "100% accurate".
