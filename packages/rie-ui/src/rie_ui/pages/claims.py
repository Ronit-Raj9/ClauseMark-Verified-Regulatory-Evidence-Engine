"""Claims page — filterable list of all Layer-1 claims."""

from __future__ import annotations

import pandas as pd
from rie_contracts.models import Layer1Status

from rie_ui.api_client import ApiClient, ApiError, ClaimSummary
from rie_ui.state import ClaimsFilter, get_filter, set_active_claim, set_filter


def _claims_to_df(claims: list[ClaimSummary]) -> pd.DataFrame:
    if not claims:
        return pd.DataFrame(
            columns=[
                "claim_id",
                "jurisdiction",
                "pillar_id",
                "indicator_id",
                "clause_id",
                "layer1_status",
                "verification_status",
                "snippet",
            ]
        )
    rows = [
        {
            "claim_id": c.claim_id,
            "jurisdiction": c.jurisdiction,
            "pillar_id": c.pillar_id,
            "indicator_id": c.indicator_id,
            "clause_id": c.clause_id,
            "layer1_status": c.layer1_status.value,
            "verification_status": (c.verification_status.value if c.verification_status else "—"),
            "snippet": c.snippet,
        }
        for c in claims
    ]
    return pd.DataFrame(rows)


def render(api: ApiClient | None = None) -> None:
    """Streamlit entrypoint for the Claims page."""
    import streamlit as st

    client = api or ApiClient()
    st.title("Claims")
    st.caption("Every Layer-1 claim with its verification status. Click a row to inspect.")

    current = get_filter()
    with st.expander("Filters", expanded=True):
        cols = st.columns(4)
        jurisdiction = cols[0].text_input("Jurisdiction", value=current.jurisdiction or "")
        pillar_id = cols[1].text_input("Pillar", value=current.pillar_id or "")
        indicator_id = cols[2].text_input("Indicator", value=current.indicator_id or "")
        status_choice = cols[3].selectbox(
            "Layer-1 status",
            options=["", *[s.value for s in Layer1Status]],
            index=(
                0
                if current.status is None
                else [s.value for s in Layer1Status].index(current.status.value) + 1
            ),
        )
        apply = st.button("Apply filters", type="primary")
        if apply:
            new_filter = ClaimsFilter(
                jurisdiction=jurisdiction.strip() or None,
                pillar_id=pillar_id.strip() or None,
                indicator_id=indicator_id.strip() or None,
                status=Layer1Status(status_choice) if status_choice else None,
            )
            set_filter(new_filter)
            current = new_filter

    try:
        claims = client.list_claims(
            jurisdiction=current.jurisdiction,
            pillar_id=current.pillar_id,
            indicator_id=current.indicator_id,
            status=current.status,
        )
    except ApiError as exc:
        st.error(f"Failed to fetch claims: {exc}")
        return

    df = _claims_to_df(claims)
    st.caption(f"{len(df)} claim(s) matching filter.")

    if df.empty:
        st.info("No claims match the current filter.")
        return

    selection = st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
    )
    selected_rows = getattr(selection, "selection", {}).get("rows", [])
    if selected_rows:
        idx = int(selected_rows[0])
        claim_id = str(df.iloc[idx]["claim_id"])
        set_active_claim(claim_id)
        st.success(f"Selected claim `{claim_id}` — open the *Claim detail* page.")


__all__ = ["render"]
