"""Filesystem-backed implementation of `ConfigRepositoryPort`.

Loads + JSON-schema-validates pillar configs, source registries, and gold sets
from on-disk YAML. Fails LOUD on schema violation — no silent skips.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012
from rie_contracts import (
    GoldItem,
    PillarConfig,
    RegistryEntry,
    SourceRegistryEntry,
)


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class _SchemaPaths:
    pillar: Path
    indicator: Path
    source_registry: Path
    gold_item: Path


def _schema_paths(repo_root: Path) -> _SchemaPaths:
    return _SchemaPaths(
        pillar=repo_root / "pillars" / "_schema" / "pillar.schema.json",
        indicator=repo_root / "pillars" / "_schema" / "indicator.schema.json",
        source_registry=repo_root / "sources" / "_schema" / "source_registry.schema.json",
        gold_item=repo_root / "gold" / "_schema" / "gold_item.schema.json",
    )


@cache
def _load_schema_registry(repo_root: Path) -> Registry:
    paths = _schema_paths(repo_root)
    pillar_schema = json.loads(paths.pillar.read_text())
    indicator_schema = json.loads(paths.indicator.read_text())
    registry: Registry = Registry().with_resources(
        [
            (pillar_schema["$id"], Resource(contents=pillar_schema, specification=DRAFT202012)),
            (
                indicator_schema["$id"],
                Resource(contents=indicator_schema, specification=DRAFT202012),
            ),
            (
                "indicator.schema.json",
                Resource(contents=indicator_schema, specification=DRAFT202012),
            ),
        ]
    )
    return registry


def _validate(data: Any, schema: dict[str, Any], registry: Registry | None = None) -> None:
    validator = (
        Draft202012Validator(schema, registry=registry)
        if registry
        else Draft202012Validator(schema)
    )
    errors = sorted(validator.iter_errors(data), key=lambda e: list(e.path))
    if errors:
        msg = "; ".join(f"{'/'.join(map(str, e.path))}: {e.message}" for e in errors)
        raise ConfigError(msg)


def _load_yaml(path: Path) -> Any:
    if not path.exists():
        raise ConfigError(f"missing config file: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@dataclass
class ConfigRepository:
    """Implements `ConfigRepositoryPort`. Construct with the repo root."""

    repo_root: Path

    # ─── Registry ───────────────────────────────────────────────────────────

    def load_registry(self) -> Sequence[RegistryEntry]:
        registry_path = self.repo_root / "pillars" / "registry.yaml"
        data = _load_yaml(registry_path)
        if not isinstance(data, dict) or "pillars" not in data:
            raise ConfigError(f"{registry_path}: missing top-level 'pillars' key")
        out: list[RegistryEntry] = []
        for raw in data["pillars"]:
            out.append(RegistryEntry.model_validate(raw))
        return out

    # ─── Pillar configs ─────────────────────────────────────────────────────

    def load_pillar(self, pillar_id: str) -> PillarConfig:
        entry = next((e for e in self.load_registry() if e.pillar_id == pillar_id), None)
        if entry is None:
            raise ConfigError(f"unknown pillar_id {pillar_id!r}")
        return self.load_pillar_from_path(self.repo_root / entry.config_path)

    def load_pillar_from_path(self, config_path: Path) -> PillarConfig:
        data = _load_yaml(config_path)
        paths = _schema_paths(self.repo_root)
        schema = json.loads(paths.pillar.read_text())
        registry = _load_schema_registry(self.repo_root)
        _validate(data, schema, registry=registry)
        try:
            return PillarConfig.model_validate(data)
        except Exception as e:
            raise ConfigError(f"{config_path}: {e}") from e

    def load_all_pillars(self, *, status_filter: set[str] | None = None) -> list[PillarConfig]:
        entries = self.load_registry()
        if status_filter:
            entries = [e for e in entries if e.status in status_filter]
        return [self.load_pillar_from_path(self.repo_root / e.config_path) for e in entries]

    # ─── Source registries ──────────────────────────────────────────────────

    def load_source_registry(self, jurisdiction: str) -> Sequence[SourceRegistryEntry]:
        path = self.repo_root / "sources" / "jurisdictions" / f"{jurisdiction.lower()}.yaml"
        data = _load_yaml(path)
        paths = _schema_paths(self.repo_root)
        schema = json.loads(paths.source_registry.read_text())
        _validate(data, schema)
        out: list[SourceRegistryEntry] = []
        for raw in data["sources"]:
            payload = dict(raw)
            payload.setdefault("jurisdiction", data["jurisdiction"])
            payload.setdefault("language", data.get("language", "en"))
            out.append(SourceRegistryEntry.model_validate(payload))
        return out

    # ─── Gold sets ─────────────────────────────────────────────────────────

    def load_gold(self, pillar_id: str) -> Sequence[GoldItem]:
        gold_dir = self.repo_root / "gold" / f"pillar_{int(pillar_id):02d}"
        if not gold_dir.exists():
            return []
        paths = _schema_paths(self.repo_root)
        item_schema = json.loads(paths.gold_item.read_text())
        items: list[GoldItem] = []
        for yml in sorted(gold_dir.glob("*.yaml")):
            data = _load_yaml(yml)
            if not isinstance(data, dict) or "items" not in data:
                raise ConfigError(f"{yml}: missing top-level 'items' key")
            for raw in data["items"]:
                _validate(raw, item_schema)
                items.append(GoldItem.model_validate(raw))
        return items
