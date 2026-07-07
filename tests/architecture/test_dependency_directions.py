"""Fitness function — enforces hexagonal dependency direction.

If this test fails, an adapter has reached across a port or `rie-domain` /
`rie-contracts` has been polluted. Fix the architecture, not the test.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKAGES = ROOT / "packages"

# Allowed workspace imports (other workspace packages this package may depend on).
ALLOWED: dict[str, set[str]] = {
    "rie_contracts": set(),
    "rie_domain": {"rie_contracts"},
    "rie_config": {"rie_contracts"},
    "rie_profiles": {"rie_contracts"},
    "rie_ingest": {"rie_contracts", "rie_config"},
    "rie_extract": {"rie_contracts", "rie_profiles"},
    "rie_retrieval": {"rie_contracts"},
    "rie_classify": {"rie_contracts"},
    "rie_verify": {"rie_contracts"},
    "rie_coverage": {"rie_contracts", "rie_domain"},
    "rie_output": {"rie_contracts"},
    "rie_persistence": {"rie_contracts"},
    "rie_orchestration": {
        "rie_contracts",
        "rie_domain",
        "rie_config",
        "rie_profiles",
        "rie_ingest",
        "rie_extract",
        "rie_retrieval",
        "rie_classify",
        "rie_verify",
        "rie_coverage",
        "rie_output",
        "rie_persistence",
    },
    "rie_api": {
        "rie_contracts",
        "rie_orchestration",
        "rie_persistence",
        "rie_config",
    },
    "rie_ui": {"rie_contracts"},  # UI calls API over HTTP, NOT in-process
    "rie_eval": {"rie_contracts", "rie_config", "rie_persistence", "rie_verify"},
}

ALL_WORKSPACE_MODULES = set(ALLOWED.keys())


def _py_files(pkg_root: Path) -> Iterator[Path]:
    for p in pkg_root.rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        yield p


def _imports(path: Path) -> Iterator[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module.split(".")[0]


@pytest.mark.parametrize("module", sorted(ALL_WORKSPACE_MODULES))
def test_dependency_direction(module: str) -> None:
    allowed = ALLOWED[module]
    src = PACKAGES / module.replace("_", "-") / "src" / module
    if not src.exists():
        pytest.skip(f"{src} missing — package not yet present")

    violations: list[str] = []
    for f in _py_files(src):
        for imp in _imports(f):
            if imp in ALL_WORKSPACE_MODULES and imp != module and imp not in allowed:
                violations.append(f"{f.relative_to(ROOT)}: imports {imp} (not allowed)")

    assert not violations, "Dependency-direction violations:\n" + "\n".join(violations)


def test_no_hardcoded_indicator_ids() -> None:
    """Engine code must NOT mention specific indicator ids — they live in pillars/."""
    import re

    # Indicator ids look like NN.M where NN is 1-12 and M is >= 1.
    # ScoreBand values ("0", "0.5", "1") are explicitly excluded.
    forbidden = re.compile(r"['\"](?:[1-9]|1[0-2])\.[1-9]\d?['\"]")
    engine_pkgs = [m for m in ALL_WORKSPACE_MODULES if m not in {"rie_eval"}]
    offenders: list[str] = []
    for module in engine_pkgs:
        src = PACKAGES / module.replace("_", "-") / "src" / module
        if not src.exists():
            continue
        for f in _py_files(src):
            text = f.read_text(encoding="utf-8")
            for match in forbidden.finditer(text):
                line = text[: match.start()].count("\n") + 1
                offenders.append(f"{f.relative_to(ROOT)}:{line}: literal {match.group()}")
    assert not offenders, "Hard-coded indicator ids in engine code:\n" + "\n".join(offenders)
