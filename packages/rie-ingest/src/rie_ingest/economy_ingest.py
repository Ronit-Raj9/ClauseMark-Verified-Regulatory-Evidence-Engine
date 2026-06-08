"""Per-economy real-document ingest.

`IngestService` (in `service.py`) implements the frozen `IngestPort`: it returns
`(DocumentMeta, bytes)` from a single `SourceRegistryEntry`. That contract is
deliberately minimal and FROZEN — it carries no `law_number_ref` / `last_amended`
fields, yet the hackathon output schema requires both columns on every row.

This module is the additive bridge. `load_economy_documents(jurisdiction)` reads
the economy's source registry (e.g. `sources/jurisdictions/singapore.yaml`,
created concurrently by the source-data agent), loads each entry's bytes through
`IngestService` (local path first, HTTP fetch with tenacity retries otherwise,
sha256-keyed cache under `data/cache/raw/`), and returns a list of

    (DocumentMeta, bytes, extra_meta)

triples. The `extra_meta` dict carries the registry-level fields the frozen
`DocumentMeta` cannot hold, so the orchestration lead can populate the output
CSV without us mutating `rie-contracts`.

NO LLM calls. NO contract mutation. Tolerant of a missing registry (clear
error) so it composes with the source-data agent's concurrent work.

`extra_meta` keys (documented for the lead — `ExtraMeta` TypedDict below):
  * ``source_id``        — registry `source_id` (== `DocumentMeta.doc_id`).
  * ``law_number_ref``   — official law number ("Act 709", "Law No. 27/2022").
                           `None` if the registry entry does not declare one.
  * ``last_amended``     — year (str) of most recent amendment; `None` if never
                           / not declared.
  * ``source_url``       — direct URL to the law on the official portal (mirror
                           of `DocumentMeta.source_url`, surfaced for the CSV).
  * ``local_path``       — the registry `local_path` if the bytes came from disk.

Where do `law_number_ref` / `last_amended` come from? `SourceRegistryEntry` is
frozen and has no such fields, so the source-data agent stashes them in an
``extra:`` mapping on each registry row. Both the JSON-schema (under
``sources/_schema/``) and the contract model forbid unknown keys, so we DO NOT
route the registry through `ConfigRepository.load_source_registry`; instead we
parse the raw YAML ourselves (`_parse_registry`), split each row's ``extra:``
block off BEFORE validating the contract-valid subset through the Pydantic
model. If the agent has not added an ``extra:`` block yet, the fields degrade to
`None` and the rest of the pipeline still runs. This keeps `rie-ingest`
independent of the source-data agent's schema choices.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import yaml
from rie_config import ConfigRepository
from rie_contracts import DocumentMeta, SourceRegistryEntry

from rie_ingest.service import IngestError, IngestService


class ExtraMeta(TypedDict):
    """Registry-level fields the frozen `DocumentMeta` cannot carry.

    Documented contract between `rie-ingest` and the orchestration lead — these
    feed the output CSV columns `law_number_ref`, `last_amended`, `source_url`.
    """

    source_id: str
    law_number_ref: str | None
    last_amended: str | None
    source_url: str | None
    local_path: str | None


# A loaded economy document: provenance, raw bytes, and the CSV-facing extras.
EconomyDocument = tuple[DocumentMeta, bytes, ExtraMeta]


class EconomyIngestError(IngestError):
    """Raised when an economy's source registry is missing or unreadable."""


def _source_registry_path(repo_root: Path, jurisdiction: str) -> Path:
    return repo_root / "sources" / "jurisdictions" / f"{jurisdiction.lower()}.yaml"


# Keys the frozen `SourceRegistryEntry` model accepts. Anything else on a
# registry row (notably the ``extra:`` block) is split off BEFORE validation so
# the contract model's `extra="forbid"` never rejects the source-data agent's
# additions. Kept in sync with `rie_contracts.SourceRegistryEntry`.
_ENTRY_FIELDS: frozenset[str] = frozenset(
    {
        "source_id",
        "jurisdiction",
        "title",
        "source_url",
        "document_type",
        "authority_tier",
        "effective_date",
        "sha256_hash",
        "language",
        "local_path",
    }
)


def _parse_registry(
    repo_root: Path, jurisdiction: str
) -> tuple[list[SourceRegistryEntry], dict[str, dict[str, object]]]:
    """Parse the raw registry YAML into validated entries + per-row extras.

    We read the YAML directly (instead of `ConfigRepository.load_source_registry`)
    because the frozen `SourceRegistryEntry` schema/model forbid unknown keys,
    yet the source-data agent stashes `law_number_ref` / `last_amended` under a
    sibling ``extra:`` mapping per source row. We split that off here, validate
    the remaining contract-valid subset through the Pydantic model (which still
    enforces required fields + enums via `extra="forbid"`), and return the
    extras keyed by `source_id`.
    """
    path = _source_registry_path(repo_root, jurisdiction)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "sources" not in data:
        raise EconomyIngestError(f"{path}: missing top-level 'sources' key")
    default_jur = data.get("jurisdiction", jurisdiction)
    default_lang = data.get("language", "en")
    sources = data["sources"]
    if not isinstance(sources, list):
        raise EconomyIngestError(f"{path}: 'sources' must be a list")

    entries: list[SourceRegistryEntry] = []
    extras: dict[str, dict[str, object]] = {}
    for raw in sources:
        if not isinstance(raw, dict):
            raise EconomyIngestError(f"{path}: each source must be a mapping")
        extra_block = raw.get("extra")
        if isinstance(extra_block, dict):
            sid = raw.get("source_id")
            if isinstance(sid, str):
                extras[sid] = extra_block
        # Keep only contract-valid keys; the model validates types/enums.
        payload: dict[str, object] = {k: v for k, v in raw.items() if k in _ENTRY_FIELDS}
        payload.setdefault("jurisdiction", default_jur)
        payload.setdefault("language", default_lang)
        try:
            entries.append(SourceRegistryEntry.model_validate(payload))
        except Exception as exc:  # re-raise as our error type
            raise EconomyIngestError(f"{path}: invalid source row: {exc}") from exc
    return entries, extras


def _coerce_optional_str(value: object) -> str | None:
    """Render a YAML scalar as a trimmed string, or `None` if absent/blank."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _build_extra_meta(
    entry: SourceRegistryEntry,
    meta: DocumentMeta,
    extra_block: dict[str, object],
) -> ExtraMeta:
    return ExtraMeta(
        source_id=entry.source_id,
        law_number_ref=_coerce_optional_str(extra_block.get("law_number_ref")),
        last_amended=_coerce_optional_str(extra_block.get("last_amended")),
        # Prefer the materialised meta URL; fall back to the registry entry.
        source_url=meta.source_url or entry.source_url,
        local_path=entry.local_path,
    )


@dataclass
class EconomyIngestor:
    """Loads every document for one economy, with CSV-facing extra metadata.

    Thin orchestration over `IngestService` — reuses its local-path / HTTP /
    sha256-cache / tenacity-retry machinery unchanged, then layers the
    registry-level `extra` fields on top.
    """

    repo_root: Path
    config: ConfigRepository
    service: IngestService

    @classmethod
    def for_repo(cls, repo_root: Path) -> EconomyIngestor:
        """Construct with a default `ConfigRepository` + `IngestService`."""
        config = ConfigRepository(repo_root=repo_root)
        service = IngestService(repo_root=repo_root, config=config)
        return cls(repo_root=repo_root, config=config, service=service)

    def load_economy_documents(self, jurisdiction: str) -> list[EconomyDocument]:
        """Load all documents for `jurisdiction`.

        Returns `(DocumentMeta, bytes, ExtraMeta)` per registry entry. Raises
        `EconomyIngestError` with a clear message if the economy's source
        registry does not exist yet (the source-data agent may not have created
        it). Per-entry byte-fetch failures surface as `IngestError` from
        `IngestService`.
        """
        path = _source_registry_path(self.repo_root, jurisdiction)
        if not path.exists():
            raise EconomyIngestError(
                f"source registry for {jurisdiction!r} not found at {path}. "
                "It may not have been created yet by the source-data agent — "
                "add sources/jurisdictions/"
                f"{jurisdiction.lower()}.yaml or pass an economy that exists."
            )

        entries, extra_blocks = _parse_registry(self.repo_root, jurisdiction)

        out: list[EconomyDocument] = []
        for entry in entries:
            meta, raw = self.service.load_document_bytes(entry)
            extra = _build_extra_meta(entry, meta, extra_blocks.get(entry.source_id, {}))
            out.append((meta, raw, extra))
        return out


def load_economy_documents(repo_root: Path, jurisdiction: str) -> list[EconomyDocument]:
    """Module-level convenience: build an `EconomyIngestor` and load.

    Returns `(DocumentMeta, bytes, ExtraMeta)` triples. See `ExtraMeta` for the
    extra-dict key contract handed to the orchestration lead.
    """
    return EconomyIngestor.for_repo(repo_root).load_economy_documents(jurisdiction)


def available_economies(repo_root: Path) -> Sequence[str]:
    """List jurisdictions with a source registry on disk (lower-cased stems).

    Useful for a `batch_run` over every economy the source-data agent has
    populated, while tolerating ones not created yet.
    """
    jur_dir = repo_root / "sources" / "jurisdictions"
    if not jur_dir.exists():
        return []
    stems = sorted(
        p.stem for p in jur_dir.glob("*.yaml") if p.is_file() and not p.stem.startswith("_")
    )
    return stems
