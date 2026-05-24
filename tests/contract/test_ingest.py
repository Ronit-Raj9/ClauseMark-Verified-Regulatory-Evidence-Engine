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
from rie_ingest import IngestService, compute_sha256

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
