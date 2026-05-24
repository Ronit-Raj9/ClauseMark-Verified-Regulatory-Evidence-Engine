"""Claim detail page — side-by-side document + decomposition + review form."""

from __future__ import annotations

from rie_contracts.models import (
    ClausePattern,
    Layer1Status,
    ReviewDecision,
    ScoreBand,
    VerificationStatus,
)

from rie_ui.api_client import ApiClient, ApiError, ClaimDetail
from rie_ui.highlight import render_with_highlight
from rie_ui.state import get_active_claim, get_reviewer, set_reviewer

# Visual chrome for the highlighted document.
_HIGHLIGHT_CSS = """
<style>
.rie-doc {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.85rem;
  white-space: pre-wrap;
  border: 1px solid var(--secondary-background-color, #eee);
  padding: 0.75rem;
  border-radius: 6px;
  max-height: 70vh;
  overflow-y: auto;
}
.rie-doc mark {
  background: #fff3a3;
  padding: 0 1px;
  border-radius: 2px;
}
</style>
"""


def _verification_badge(status: VerificationStatus | None) -> str:
    if status is None:
        return ":grey_question: unverified"
    return {
        VerificationStatus.VERIFIED: ":white_check_mark: verified",
        VerificationStatus.FLAGGED: ":warning: flagged",
        VerificationStatus.REJECTED: ":x: rejected",
    }[status]


def _layer1_badge(status: Layer1Status) -> str:
    return {
        Layer1Status.PENDING_VERIFICATION: ":hourglass: pending",
        Layer1Status.VERIFIED: ":white_check_mark: verified",
        Layer1Status.FLAGGED: ":warning: flagged",
        Layer1Status.REJECTED: ":x: rejected",
    }[status]


def _render_document(st_mod, detail: ClaimDetail) -> None:
    spans = [(s.char_start, s.char_end) for s in detail.claim.evidence_spans]
    html = render_with_highlight(detail.document_text, spans)
    st_mod.markdown(_HIGHLIGHT_CSS, unsafe_allow_html=True)
    st_mod.markdown(
        f"**Document:** `{detail.document_id}` — {detail.document_title or '(untitled)'}"
    )
    st_mod.caption(f"{len(detail.claim.evidence_spans)} cited span(s).")
    st_mod.markdown(f'<div class="rie-doc">{html}</div>', unsafe_allow_html=True)


def _render_decomposition(st_mod, detail: ClaimDetail) -> None:
    claim = detail.claim
    st_mod.markdown(f"### Claim · `{claim.claim_id}`")
    cols = st_mod.columns(2)
    cols[0].markdown(f"**Indicator:** `{claim.indicator_id}`")
    cols[1].markdown(f"**Pillar:** `{claim.pillar_id}`")

    cols = st_mod.columns(2)
    cols[0].markdown(f"**Pattern:** `{ClausePattern(claim.clause_pattern).value}`")
    cols[1].markdown(f"**Layer-1:** {_layer1_badge(claim.layer1_status)}")

    st_mod.markdown("**Decomposition**")
    st_mod.json(claim.decomposition.model_dump(), expanded=True)

    if detail.verification is not None:
        st_mod.markdown(f"**Verification:** {_verification_badge(detail.verification.status)}")
        with st_mod.expander("Gate results", expanded=False):
            st_mod.json(
                [g.model_dump(mode="json") for g in detail.verification.gates],
                expanded=False,
            )
        if detail.verification.failure_reasons:
            st_mod.warning("\n".join(detail.verification.failure_reasons))

    if detail.layer2 is not None:
        st_mod.markdown("**Layer-2 recommendation (human confirmation required)**")
        st_mod.info(
            f"Recommended band: **{detail.layer2.recommended_band.value}** — "
            f"{detail.layer2.rationale}"
        )
        if detail.layer2.open_questions:
            st_mod.markdown("Open questions:")
            for q in detail.layer2.open_questions:
                st_mod.markdown(f"- {q}")

    if detail.reviews:
        with st_mod.expander(f"Previous reviews ({len(detail.reviews)})", expanded=False):
            for review in detail.reviews:
                st_mod.markdown(
                    f"- **{review.reviewer}** · `{review.decision.value}` · "
                    f"{review.decided_at:%Y-%m-%d %H:%M} — {review.note or '(no note)'}"
                )


def _render_review_form(st_mod, detail: ClaimDetail, client: ApiClient) -> None:
    st_mod.markdown("### Reviewer decision")
    reviewer = st_mod.text_input("Reviewer", value=get_reviewer())
    set_reviewer(reviewer or "anonymous")

    note = st_mod.text_area("Note (optional)", height=80, key="rie-review-note")
    needs_score = st_mod.checkbox(
        "Provide corrected score band", value=False, key="rie-review-needs-score"
    )
    corrected: ScoreBand | None = None
    if needs_score:
        choice = st_mod.selectbox(
            "Corrected band",
            options=[b.value for b in ScoreBand],
            index=0,
            key="rie-review-score",
        )
        corrected = ScoreBand(choice)

    cols = st_mod.columns(3)
    decision: ReviewDecision | None = None
    if cols[0].button("Accept", type="primary", use_container_width=True):
        decision = ReviewDecision.ACCEPT
    if cols[1].button("Correct", use_container_width=True):
        decision = ReviewDecision.CORRECT
    if cols[2].button("Reject", use_container_width=True):
        decision = ReviewDecision.REJECT

    if decision is None:
        return

    try:
        record = client.submit_review(
            claim_id=detail.claim.claim_id,
            reviewer=reviewer or "anonymous",
            decision=decision,
            corrected_score=corrected,
            note=note,
        )
    except ApiError as exc:
        st_mod.error(f"Failed to submit review: {exc}")
        return

    st_mod.success(
        f"Recorded **{record.decision.value}** by *{record.reviewer}* at "
        f"{record.decided_at:%Y-%m-%d %H:%M}."
    )


def render(api: ApiClient | None = None) -> None:
    """Streamlit entrypoint for the Claim detail page."""
    import streamlit as st

    client = api or ApiClient()
    st.title("Claim detail")

    claim_id = get_active_claim()
    if not claim_id:
        st.info("Pick a claim from the *Claims* page first.")
        return

    try:
        detail = client.get_claim(claim_id)
    except ApiError as exc:
        st.error(f"Failed to fetch claim `{claim_id}`: {exc}")
        return

    left, right = st.columns([3, 2], gap="large")
    with left:
        _render_document(st, detail)
    with right:
        _render_decomposition(st, detail)
        st.divider()
        _render_review_form(st, detail, client)


__all__ = ["render"]
