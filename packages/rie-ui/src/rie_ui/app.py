"""Streamlit entrypoint — multi-page RIE audit viewer.

Run with::

    streamlit run packages/rie-ui/src/rie_ui/app.py

``build_app()`` is a thin builder used by the contract test and as the body of
``main()``. Streamlit invocations of the script land in ``main()`` which calls
``build_app()`` and dispatches to the chosen page.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from rie_ui.api_client import DEFAULT_BASE_URL, ApiClient
from rie_ui.pages import audit, claim_detail, claims, coverage, run


@dataclass(frozen=True, slots=True)
class PageSpec:
    """A page in the sidebar — title plus its ``render(api)`` callable."""

    key: str
    title: str
    icon: str
    render: Callable[[ApiClient], None]


def _pages() -> list[PageSpec]:
    return [
        PageSpec("run", "Run", ":rocket:", run.render),
        PageSpec("claims", "Claims", ":mag:", claims.render),
        PageSpec("claim_detail", "Claim detail", ":scroll:", claim_detail.render),
        PageSpec("coverage", "Coverage", ":world_map:", coverage.render),
        PageSpec("audit", "Audit package", ":package:", audit.render),
    ]


@dataclass(frozen=True, slots=True)
class AppContext:
    """Bundle of objects a page needs — the API client and the page registry."""

    api: ApiClient
    pages: tuple[PageSpec, ...]


def build_app(api: ApiClient | None = None) -> AppContext:
    """Create the page registry + API client.

    Pure function — no Streamlit calls. The contract test exercises this to
    prove the package imports cleanly without spinning up a Streamlit runtime.
    """
    client = api or ApiClient()
    return AppContext(api=client, pages=tuple(_pages()))


def _render_sidebar(st_mod, ctx: AppContext) -> PageSpec:
    st_mod.sidebar.title("RIE Audit Viewer")
    st_mod.sidebar.caption(
        "Pillar-agnostic evidence-extraction & verification engine — reviewer view."
    )
    titles = [f"{p.icon} {p.title}" for p in ctx.pages]
    choice = st_mod.sidebar.radio("Page", titles, index=0, label_visibility="collapsed")
    idx = titles.index(choice)
    st_mod.sidebar.divider()
    st_mod.sidebar.caption(f"API: `{ctx.api.base_url}`")
    st_mod.sidebar.caption(f"(default `{DEFAULT_BASE_URL}`)")
    return ctx.pages[idx]


def main() -> None:
    """Streamlit script body."""
    import streamlit as st

    st.set_page_config(
        page_title="RIE Audit Viewer",
        page_icon=":balance_scale:",
        layout="wide",
    )
    ctx = build_app()
    page = _render_sidebar(st, ctx)
    page.render(ctx.api)


# Streamlit imports this file as a script; the guard keeps ``import rie_ui.app``
# side-effect-free.
if __name__ == "__main__":  # pragma: no cover - executed by streamlit run
    main()


__all__ = ["AppContext", "PageSpec", "build_app", "main"]
