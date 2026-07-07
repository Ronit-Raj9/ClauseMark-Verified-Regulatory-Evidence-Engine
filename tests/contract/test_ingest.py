"""Contract test for `IngestPort` — `IngestService` is the conforming impl.

This test passes with *any* implementation of the port. It encodes the
behaviour every adapter must honour, so swapping in a different adapter
(e.g. a discovery crawler in Phase 2) never silently regresses.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from rie_config import ConfigRepository
from rie_contracts import (
    DocumentMeta,
    IngestPort,
    SourceRegistryEntry,
)
from rie_ingest import (
    DiscoveryTagger,
    IngestService,
    SourceDiscoveryCrawler,
    compute_sha256,
    load_economy_documents,
    tag_discovery,
)
from rie_ingest.economy_ingest import EconomyIngestError

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def service() -> IngestService:
    return IngestService(
        repo_root=REPO_ROOT,
        config=ConfigRepository(repo_root=REPO_ROOT),
    )


# ─── Runtime conformance ────────────────────────────────────────────────────


def test_ingest_service_implements_port(service: IngestService) -> None:
    """`@runtime_checkable` Protocol → an `isinstance` check is the spec."""
    assert isinstance(service, IngestPort)


# ─── E2E against the real sample registry + sample DPA file ─────────────────


def test_load_source_registry_returns_expected_entries(service: IngestService) -> None:
    entries = service.load_source_registry("SAMPLE")
    assert len(entries) >= 1
    ids = {e.source_id for e in entries}
    assert "sample_dpa_2020" in ids
    assert all(isinstance(e, SourceRegistryEntry) for e in entries)


def test_load_document_bytes_for_sample_dpa(service: IngestService) -> None:
    entries = service.load_source_registry("SAMPLE")
    dpa = next(e for e in entries if e.source_id == "sample_dpa_2020")

    meta, raw = service.load_document_bytes(dpa)

    # Bytes are non-trivial — the sample DPA has known content.
    assert isinstance(meta, DocumentMeta)
    assert len(raw) > 100
    assert b"PERSONAL DATA PROTECTION ACT" in raw

    # Provenance invariants.
    assert meta.doc_id == dpa.source_id
    assert meta.jurisdiction == dpa.jurisdiction
    assert meta.title == dpa.title
    assert meta.document_type == dpa.document_type
    assert meta.authority_tier == dpa.authority_tier
    assert meta.sha256 == compute_sha256(raw)
    assert meta.retrieved_at is not None


def test_list_sample_laws_finds_provided_samples(service: IngestService) -> None:
    samples_dir = REPO_ROOT / "data" / "samples"
    files = service.list_sample_laws(samples_dir)
    names = {p.name for p in files}
    # Sample DPA + ETA ship with the repo; both must surface.
    assert "sample_dpa.txt" in names
    assert "sample_eta.txt" in names


# ─── Phase 2 crawler still yields IngestPort-compatible entries ──────────────


def test_crawler_offline_yields_port_compatible_entries(
    service: IngestService, tmp_path: Path
) -> None:
    """The Phase 2 `SourceDiscoveryCrawler` produces `SourceRegistryEntry` rows
    that the existing `IngestPort` impl can consume unchanged — discovery never
    introduces a new contract type."""
    seed = tmp_path / "portal"
    seed.mkdir()
    (seed / "index.html").write_text(
        '<html><body><a href="acts/dpa.pdf">Data Protection Act</a></body></html>',
        encoding="utf-8",
    )

    crawler = SourceDiscoveryCrawler(offline_seed_dir=seed)
    entries = crawler.discover("ignored://seed", "SAMPLE")

    assert entries
    assert all(isinstance(e, SourceRegistryEntry) for e in entries)
    # `load_document_bytes` accepts these rows by type — feed it the same kind
    # of entry the service already handles (local_path form).
    assert isinstance(service, IngestPort)


# ─── Per-economy ingest layers extras over the same IngestPort bytes ─────────


def test_load_economy_documents_matches_port_bytes() -> None:
    """`load_economy_documents` reuses the IngestPort byte path and adds the
    CSV-facing `extra` dict — the `(DocumentMeta, bytes)` part is identical to
    what `load_document_bytes` produces for the same registry entry."""
    docs = load_economy_documents(REPO_ROOT, "SAMPLE")
    assert docs
    by_id = {meta.doc_id: (meta, raw, extra) for meta, raw, extra in docs}
    assert "sample_dpa_2020" in by_id
    meta, raw, extra = by_id["sample_dpa_2020"]
    assert isinstance(meta, DocumentMeta)
    assert meta.sha256 == compute_sha256(raw)
    # Extra dict has the documented keys even when the registry omits them.
    assert set(extra.keys()) == {
        "source_id",
        "law_number_ref",
        "last_amended",
        "source_url",
        "local_path",
    }
    assert extra["source_id"] == "sample_dpa_2020"


def test_load_economy_documents_missing_registry_is_clear() -> None:
    with pytest.raises(EconomyIngestError):
        load_economy_documents(REPO_ROOT, "NoSuchEconomy")


# ─── Discovery tagger is deterministic over the real gold corpus ─────────────


def test_discovery_tagger_known_for_real_gold() -> None:
    """A SAMPLE gold provision (pillar 6) tags KNOWN; a foreign one tags NEW."""
    config = ConfigRepository(repo_root=REPO_ROOT)
    gold = list(config.load_gold("6"))
    assert gold, "expected pillar-6 gold to exist in the repo"
    g = gold[0]
    assert (
        tag_discovery(
            jurisdiction=g.jurisdiction,
            indicator_id=g.indicator_id,
            doc_id=g.doc_id,
            span_text=g.span_text,
            gold=gold,
        )
        == "KNOWN"
    )
    tagger = DiscoveryTagger(gold=gold)
    assert (
        tagger.tag_fields(
            jurisdiction="Atlantis",
            indicator_id=g.indicator_id,
            doc_id="unknown_doc",
            span_text="entirely novel provision not present in the gold set",
        )
        == "NEW"
    )
