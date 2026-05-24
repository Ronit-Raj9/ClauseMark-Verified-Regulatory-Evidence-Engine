"""Run page — start a new ingestion+classification run, poll its status."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from rie_ui.api_client import ApiClient, ApiError, RunStatusResponse
from rie_ui.state import get_active_run, set_active_run

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass


_TERMINAL_STATES: frozenset[str] = frozenset({"succeeded", "failed", "cancelled"})


def _status_emoji(status: str) -> str:
    return {
        "pending": ":hourglass_flowing_sand:",
        "running": ":arrows_counterclockwise:",
        "succeeded": ":white_check_mark:",
        "failed": ":x:",
        "cancelled": ":no_entry_sign:",
    }.get(status, ":grey_question:")


def _render_status(st_mod, run: RunStatusResponse) -> None:
    cols = st_mod.columns([1, 1, 1, 2])
    cols[0].metric("Run ID", run.run_id[:12])
    cols[1].metric("Jurisdiction", run.jurisdiction)
    cols[2].metric("Status", f"{_status_emoji(run.status)} {run.status}")
    cols[3].write(f"Pillars: `{', '.join(run.pillar_ids) or '-'}`")

    if run.progress:
        st_mod.json(run.progress, expanded=False)
    if run.error:
        st_mod.error(run.error)


def render(api: ApiClient | None = None) -> None:
    """Streamlit entrypoint for the Run page."""
    import streamlit as st

    client = api or ApiClient()
    st.title("Start a run")
    st.caption(
        "Pick a jurisdiction and one or more pillars. The orchestrator ingests "
        "the registered sample laws, classifies clauses, and runs the 4-gate "
        "verification."
    )

    with st.form("rie-run-form", clear_on_submit=False):
        jurisdiction = st.text_input(
            "Jurisdiction (ISO code or short name)",
            value="SG",
            help="e.g. SG, EU, KR — must match a jurisdiction in the source registry.",
        )
        pillars_raw = st.text_input(
            "Pillar IDs (comma-separated)",
            value="6,7",
            help="MVP ships Pillars 6 and 7; others are roadmap.",
        )
        submitted = st.form_submit_button("Start run", type="primary")

    if submitted:
        pillar_ids = [p.strip() for p in pillars_raw.split(",") if p.strip()]
        if not jurisdiction.strip() or not pillar_ids:
            st.warning("Both jurisdiction and at least one pillar are required.")
        else:
            try:
                run = client.start_run(jurisdiction.strip(), pillar_ids)
            except ApiError as exc:
                st.error(f"Failed to start run: {exc}")
            else:
                set_active_run(run.run_id)
                st.success(f"Started run `{run.run_id}`")
                _render_status(st, run)

    active_id = get_active_run()
    if not active_id:
        st.info("No active run. Submit the form above to start one.")
        return

    st.divider()
    st.subheader(f"Active run · `{active_id}`")
    placeholder = st.empty()
    poll = st.toggle("Auto-poll status", value=False)

    try:
        run = client.get_run(active_id)
    except ApiError as exc:
        st.error(f"Failed to fetch run: {exc}")
        return

    with placeholder.container():
        _render_status(st, run)

    if poll and run.status not in _TERMINAL_STATES:
        # One light poll per render — Streamlit reruns the script which keeps
        # the loop responsive without blocking the event loop.
        time.sleep(2.0)
        st.rerun()


__all__ = ["render"]
