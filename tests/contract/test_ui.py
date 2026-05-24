"""Contract test for the rie-ui package.

We deliberately do NOT spin up Streamlit. The contract is just:
- ``rie_ui.app.build_app`` is importable and returns an :class:`AppContext`.
- ``rie_ui.api_client.ApiClient`` is importable and constructable.
- The page registry covers the five mandated pages.
"""

from __future__ import annotations

import importlib

import pytest


def test_build_app_is_importable() -> None:
    mod = importlib.import_module("rie_ui.app")
    assert hasattr(mod, "build_app"), "rie_ui.app must expose build_app()"
    ctx = mod.build_app()
    assert ctx.api is not None
    assert ctx.pages, "build_app() returned no pages"


def test_api_client_is_importable_and_constructable() -> None:
    mod = importlib.import_module("rie_ui.api_client")
    api_cls = mod.ApiClient
    api = api_cls("http://example.invalid:8080")
    try:
        assert api.base_url == "http://example.invalid:8080"
    finally:
        api.close()


def test_page_registry_covers_all_required_pages() -> None:
    from rie_ui.app import build_app

    ctx = build_app()
    keys = {p.key for p in ctx.pages}
    required = {"run", "claims", "claim_detail", "coverage", "audit"}
    missing = required - keys
    assert not missing, f"missing pages in registry: {missing}"


def test_every_page_render_is_callable() -> None:
    from rie_ui.app import build_app

    for p in build_app().pages:
        assert callable(p.render), f"page {p.key!r} has non-callable render"


def test_highlight_is_pure_and_importable() -> None:
    mod = importlib.import_module("rie_ui.highlight")
    fn = mod.render_with_highlight
    assert fn("hello", [(0, 5)]) == "<mark>hello</mark>"


def test_state_helpers_use_typed_filter() -> None:
    from rie_ui.state import ClaimsFilter, get_filter, set_filter

    fake_state: dict[str, object] = {}
    f = ClaimsFilter(jurisdiction="SG")
    set_filter(f, state=fake_state)
    got = get_filter(state=fake_state)
    assert got.jurisdiction == "SG"


def test_ui_does_not_import_orchestration_or_persistence() -> None:
    """Architectural guard restated at the contract level."""
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "packages" / "rie-ui" / "src" / "rie_ui"
    if not root.exists():  # pragma: no cover - defensive
        pytest.skip("rie-ui not present")

    banned = {"rie_orchestration", "rie_persistence", "rie_api"}
    offenders: list[str] = []
    for py in root.rglob("*.py"):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in banned:
                        offenders.append(f"{py.name}: {alias.name}")
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module.split(".")[0] in banned:
                    offenders.append(f"{py.name}: {node.module}")
    assert not offenders, "rie-ui must not import in-process: " + ", ".join(offenders)
