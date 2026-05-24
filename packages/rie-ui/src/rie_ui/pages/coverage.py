"""Coverage page — per-indicator state with most-risky rows pinned to the top."""

from __future__ import annotations

import pandas as pd
from rie_contracts.models import CoverageRecord, CoverageState

from rie_ui.api_client import ApiClient, ApiError
from rie_ui.state import get_jurisdiction, set_jurisdiction

# Sort key: lower = higher in the list. We pin "no evidence" and "insufficient
# coverage" above "evidence_found" because they carry the most hidden risk
# (§7 of the architecture).
_RISK_ORDER: dict[str, int] = {
    CoverageState.NO_EVIDENCE_IN_SEARCHED_CORPUS.value: 0,
    CoverageState.INSUFFICIENT_COVERAGE.value: 1,
    CoverageState.EVIDENCE_FOUND.value: 2,
}


def _records_to_df(records: list[CoverageRecord]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(
            columns=[
                "jurisdiction",
                "indicator_id",
                "state",
                "measured_recall",
                "reason",
                "verified_claims",
            ]
        )
    rows = [
        {
            "jurisdiction": r.jurisdiction,
            "indicator_id": r.indicator_id,
            "state": r.state.value,
            "measured_recall": r.measured_recall,
            "reason": r.reason or "",
            "verified_claims": len(r.verified_claim_ids),
            "_risk": _RISK_ORDER.get(r.state.value, 99),
        }
        for r in records
    ]
    df = pd.DataFrame(rows)
    df = df.sort_values(by=["_risk", "indicator_id"], ascending=[True, True], kind="stable")
    return df.drop(columns=["_risk"]).reset_index(drop=True)


def render(api: ApiClient | None = None) -> None:
    """Streamlit entrypoint for the Coverage page."""
    import streamlit as st

    client = api or ApiClient()
    st.title("Coverage")
    st.caption(
        "Per-indicator state. Rows with `no_evidence_in_searched_corpus` and "
        "`insufficient_coverage` are pinned to the top — they carry the most "
        "hidden risk (architecture §7)."
    )

    current_juris = get_jurisdiction()
    juris = st.text_input(
        "Jurisdiction (leave blank for all)",
        value=current_juris or "",
    )
    if juris.strip() != (current_juris or ""):
        set_jurisdiction(juris.strip() or None)

    try:
        records = client.list_coverage(jurisdiction=juris.strip() or None)
    except ApiError as exc:
        st.error(f"Failed to fetch coverage: {exc}")
        return

    df = _records_to_df(records)
    if df.empty:
        st.info("No coverage records yet for this jurisdiction.")
        return

    # Highlight the risky rows so the eye lands on them first.
    def _style_row(row: pd.Series) -> list[str]:
        if row["state"] == CoverageState.NO_EVIDENCE_IN_SEARCHED_CORPUS.value:
            return ["background-color: #fff3a3"] * len(row)
        if row["state"] == CoverageState.INSUFFICIENT_COVERAGE.value:
            return ["background-color: #ffd6d6"] * len(row)
        return [""] * len(row)

    styled = df.style.apply(_style_row, axis=1).format(
        {"measured_recall": lambda v: "—" if v is None else f"{v:.2f}"}
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)


__all__ = ["render"]
