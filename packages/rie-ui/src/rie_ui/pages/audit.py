"""Audit package page — download the full JSON evidence pack for a jurisdiction."""

from __future__ import annotations

import json

from rie_ui.api_client import ApiClient, ApiError
from rie_ui.state import get_jurisdiction, set_jurisdiction


def render(api: ApiClient | None = None) -> None:
    """Streamlit entrypoint for the Audit page."""
    import streamlit as st

    client = api or ApiClient()
    st.title("Audit package")
    st.caption(
        "Download the full machine-readable audit package for a jurisdiction. "
        "Each claim carries its verified spans, regime members, gate results, "
        "and Layer-2 recommendation."
    )

    current = get_jurisdiction()
    juris = st.text_input("Jurisdiction", value=current or "SG").strip()
    if juris and juris != (current or ""):
        set_jurisdiction(juris)

    if not juris:
        st.info("Enter a jurisdiction to fetch its audit package.")
        return

    if not st.button("Fetch audit package", type="primary"):
        return

    try:
        package = client.get_audit_package(juris)
    except ApiError as exc:
        st.error(f"Failed to fetch audit package: {exc}")
        return

    payload = json.dumps(package, indent=2, default=str).encode("utf-8")
    st.success(f"Loaded {len(payload):,} bytes.")
    st.download_button(
        "Download JSON",
        data=payload,
        file_name=f"rie_audit_{juris}.json",
        mime="application/json",
        type="primary",
    )
    with st.expander("Preview", expanded=False):
        st.json(package, expanded=False)


__all__ = ["render"]
