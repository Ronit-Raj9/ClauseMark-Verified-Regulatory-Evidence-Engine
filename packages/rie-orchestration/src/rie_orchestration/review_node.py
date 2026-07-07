"""HITL review node — pauses graph for FLAGGED claims via LangGraph `interrupt()`.

Per `tool.md`: a node containing `interrupt()` re-executes from the start on
resume. So this node does NOTHING else — it only pauses for review of each
FLAGGED claim and applies the human decision. The verification has already
been recorded by `verify_node`.

The resumed payload is a `dict[claim_id, ReviewDecision]` produced by the
audit UI / API. Accepted claims flip their stored `layer1_status` to
`VERIFIED`; corrected/rejected stay `FLAGGED` until the reviewer also writes
a `ReviewRecord` via the dedicated API.
"""

from __future__ import annotations

import logging

from rie_contracts import Layer1Status, VerificationStatus

from rie_orchestration.state import RieState
from rie_orchestration.wiring import AdapterBundle

log = logging.getLogger(__name__)


def review_node(state: RieState, bundle: AdapterBundle) -> RieState:
    """Pause graph for every FLAGGED claim. Resumes with reviewer decisions.

    Skipped entirely when no claims are FLAGGED — graph runs through.
    Also skipped when `state["skip_hitl"] is True` (set by API tests +
    `run_pipeline(dry_run=True)`).
    """
    flagged = [
        c
        for c in state.get("claims", [])
        if c.layer1_status != Layer1Status.VERIFIED
        and state.get("verifications", {}).get(c.claim_id)
        and state["verifications"][c.claim_id].status == VerificationStatus.FLAGGED
    ]
    if not flagged:
        return state
    if state.get("skip_hitl", False):
        log.info(
            "review_node: %d FLAGGED claims, skip_hitl=True — skipping interrupt", len(flagged)
        )
        return state

    try:
        from langgraph.types import interrupt
    except ImportError:
        log.warning("langgraph.types.interrupt unavailable — auto-skipping HITL")
        return state

    payload = {
        "kind": "human_review_required",
        "run_id": state.get("run_id"),
        "jurisdiction": state.get("jurisdiction"),
        "flagged_claim_ids": [c.claim_id for c in flagged],
        "instructions": "Submit a dict[claim_id, ReviewDecision] via Command(resume=...).",
    }
    log.info("review_node: interrupt() with %d FLAGGED claims", len(flagged))

    decisions: dict[str, str] = interrupt(payload)  # type: ignore[assignment]
    if not isinstance(decisions, dict):
        log.warning("review_node: unexpected resume payload type %s — skipping", type(decisions))
        return state

    accepted = 0
    for claim in flagged:
        verdict = decisions.get(claim.claim_id, "").lower()
        if verdict == "accept":
            promoted = claim.model_copy(update={"layer1_status": Layer1Status.VERIFIED})
            bundle.repo.save_claim(promoted)
            accepted += 1
            for i, c in enumerate(state["claims"]):
                if c.claim_id == claim.claim_id:
                    state["claims"][i] = promoted
    log.info("review_node: resumed; %d/%d claims accepted", accepted, len(flagged))
    return state
