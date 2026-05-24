"""Pillar registry sanity: every pillar listed validates against the schema and
the gold-set directory exists for built pillars."""

from __future__ import annotations

from pathlib import Path

import pytest
from rie_config import ConfigRepository

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def repo() -> ConfigRepository:
    return ConfigRepository(repo_root=REPO_ROOT)


def test_registry_lists_all_12_pillars(repo: ConfigRepository) -> None:
    entries = repo.load_registry()
    ids = {e.pillar_id for e in entries}
    assert ids == {str(i) for i in range(1, 13)}


def test_each_pillar_validates(repo: ConfigRepository) -> None:
    pillars = repo.load_all_pillars()
    assert len(pillars) == 12


def test_every_built_pillar_has_non_empty_gold_set(repo: ConfigRepository) -> None:
    for entry in repo.load_registry():
        if entry.status != "built":
            continue
        items = repo.load_gold(entry.pillar_id)
        assert items, f"pillar {entry.pillar_id} marked built but gold set empty"
