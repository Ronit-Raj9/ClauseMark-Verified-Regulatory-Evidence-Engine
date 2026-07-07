from pathlib import Path

import pytest
from rie_config import ConfigError, ConfigRepository
from rie_contracts import DocumentProfile

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def repo() -> ConfigRepository:
    return ConfigRepository(repo_root=REPO_ROOT)


def test_registry_has_all_12_pillars(repo: ConfigRepository) -> None:
    entries = repo.load_registry()
    pillar_ids = {e.pillar_id for e in entries}
    assert pillar_ids == {str(i) for i in range(1, 13)}, pillar_ids


def test_registry_status_distribution(repo: ConfigRepository) -> None:
    entries = repo.load_registry()
    built = {e.pillar_id for e in entries if e.status == "built"}
    # Built = digital-governance cluster (6/7 deep, 8/9/12 Phase-2 deepened + gold).
    # 1-5/10/11 remain honest stubs (different doc profiles, no gold yet).
    assert built == {"6", "7", "8", "9", "12"}
    stubs = {e.pillar_id for e in entries if e.status == "stub"}
    assert stubs == {"1", "2", "3", "4", "5", "10", "11"}


def test_load_pillar_06_deep(repo: ConfigRepository) -> None:
    p6 = repo.load_pillar("6")
    assert p6.pillar_id == "6"
    assert p6.status == "built"
    assert p6.document_profile == DocumentProfile.STATUTORY_LEGAL_TEXT
    ids = {i.indicator_id for i in p6.indicators}
    assert {"6.1", "6.2", "6.3", "6.4", "6.5"}.issubset(ids)
    i64 = next(i for i in p6.indicators if i.indicator_id == "6.4")
    assert i64.few_shot_examples, "6.4 must ship at least one few-shot example"


def test_all_pillars_validate(repo: ConfigRepository) -> None:
    pillars = repo.load_all_pillars()
    assert len(pillars) == 12


def test_invalid_pillar_raises(repo: ConfigRepository, tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("pillar_id: '99'\nshould_not_be_here: oops\n")
    with pytest.raises(ConfigError):
        repo.load_pillar_from_path(bad)


def test_load_source_registry_sample(repo: ConfigRepository) -> None:
    sources = repo.load_source_registry("SAMPLE")
    assert {s.source_id for s in sources} == {"sample_dpa_2020", "sample_ecommerce_2019"}


def test_load_gold_for_pillar_06(repo: ConfigRepository) -> None:
    items = repo.load_gold("6")
    assert len(items) >= 2
    assert any(i.indicator_id == "6.4" for i in items)
