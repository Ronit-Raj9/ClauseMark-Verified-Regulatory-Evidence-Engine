"""Unit tests for `IngestService` — local path, listing, cache, retries."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
import respx
from rie_config import ConfigRepository
from rie_contracts import (
    AuthorityTier,
    DocumentMeta,
    DocumentType,
    SourceRegistryEntry,
)
from rie_ingest import (
    IngestError,
    IngestService,
    compute_sha256,
)

# ─── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def repo_root(tmp_path: Path) -> Path:
    """Per-test fake repo root with the directory layout the service expects."""
    (tmp_path / "data" / "samples").mkdir(parents=True)
    (tmp_path / "data" / "cache" / "raw").mkdir(parents=True)
    (tmp_path / "sources" / "jurisdictions").mkdir(parents=True)
    return tmp_path


@pytest.fixture
def service(repo_root: Path) -> IngestService:
    return IngestService(
        repo_root=repo_root,
        config=ConfigRepository(repo_root=repo_root),
    )


def _entry_local(path_rel: str, source_id: str = "test_local") -> SourceRegistryEntry:
    return SourceRegistryEntry(
        source_id=source_id,
        jurisdiction="SAMPLE",
        title="Test Local Doc",
        source_url=None,
        document_type=DocumentType.STATUTE,
        authority_tier=AuthorityTier.TIER_1_STATUTE,
        local_path=path_rel,
    )


def _entry_url(url: str, *, sha: str | None = None) -> SourceRegistryEntry:
    return SourceRegistryEntry(
        source_id="test_remote",
        jurisdiction="SAMPLE",
        title="Test Remote Doc",
        source_url=url,
        document_type=DocumentType.STATUTE,
        authority_tier=AuthorityTier.TIER_1_STATUTE,
        local_path=None,
        sha256_hash=sha,
    )


# ─── load_document_bytes: local path ────────────────────────────────────────


def test_local_path_loads_and_computes_sha(repo_root: Path, service: IngestService) -> None:
    body = b"SAMPLE LAW BODY\nSection 1. Foo.\n"
    local = repo_root / "data" / "samples" / "doc.txt"
    local.write_bytes(body)

    entry = _entry_local("data/samples/doc.txt")
    meta, raw = service.load_document_bytes(entry)

    assert isinstance(meta, DocumentMeta)
    assert raw == body
    assert meta.sha256 == compute_sha256(body)
    assert meta.doc_id == entry.source_id
    assert meta.jurisdiction == entry.jurisdiction
    assert meta.title == entry.title
    assert meta.document_type == entry.document_type
    assert meta.authority_tier == entry.authority_tier
    assert meta.language == entry.language
    assert meta.retrieved_at.tzinfo is not None
    # retrieved_at within a sensible window — just sanity check it's "now"-ish.
    delta = abs((datetime.now(UTC) - meta.retrieved_at).total_seconds())
    assert delta < 5.0


def test_local_path_writes_cache(repo_root: Path, service: IngestService) -> None:
    body = b"cached body"
    (repo_root / "data" / "samples" / "doc.txt").write_bytes(body)
    entry = _entry_local("data/samples/doc.txt")

    meta, _ = service.load_document_bytes(entry)
    cache_file = repo_root / "data" / "cache" / "raw" / f"{meta.sha256}.bin"
    assert cache_file.exists()
    assert cache_file.read_bytes() == body


def test_local_path_missing_raises(repo_root: Path, service: IngestService) -> None:
    entry = _entry_local("data/samples/nope.txt")
    with pytest.raises(IngestError, match="missing"):
        service.load_document_bytes(entry)


# ─── list_sample_laws: deterministic sort + ext filter ──────────────────────


def test_list_sample_laws_sorted_and_filtered(repo_root: Path, service: IngestService) -> None:
    samples = repo_root / "data" / "samples"
    (samples / "z_last.txt").write_text("z")
    (samples / "a_first.pdf").write_bytes(b"%PDF-1.4")
    (samples / "m_mid.html").write_text("<html/>")
    (samples / "n_mid.htm").write_text("<html/>")
    # Non-matching ones must be filtered out.
    (samples / "ignore.md").write_text("# nope")
    (samples / "ignore.json").write_text("{}")
    # Subdirectories are NOT recursed — the list is top-level only.
    (samples / "subdir").mkdir()
    (samples / "subdir" / "deep.txt").write_text("deep")

    out = service.list_sample_laws(samples)
    names = [p.name for p in out]
    assert names == ["a_first.pdf", "m_mid.html", "n_mid.htm", "z_last.txt"]


def test_list_sample_laws_missing_dir_is_empty(tmp_path: Path, service: IngestService) -> None:
    assert list(service.list_sample_laws(tmp_path / "does_not_exist")) == []


# ─── HTTP fetch + cache + retries (respx) ───────────────────────────────────


@respx.mock
def test_http_get_caches_on_second_call(repo_root: Path) -> None:
    url = "https://example.gov/doc.pdf"
    body = b"%PDF-1.4 remote bytes"
    route = respx.get(url).mock(return_value=httpx.Response(200, content=body))

    # No `sha256_hash` on the entry → first call must hit the network; second
    # call also hits the network because we only cache by sha *after* the
    # fetch, and we cannot key the cache without knowing the sha up front.
    # To exercise the "skip network because cache hit" path, supply the sha.
    sha = compute_sha256(body)
    entry = _entry_url(url, sha=sha)
    # Prime the cache by simulating a previous successful run.
    service = IngestService(
        repo_root=repo_root,
        config=ConfigRepository(repo_root=repo_root),
    )
    service.cache.store(sha, body)

    meta, raw = service.load_document_bytes(entry)
    assert raw == body
    assert meta.sha256 == sha
    assert route.call_count == 0, "cached bytes must not hit the network"


@respx.mock
def test_http_get_no_prior_cache_does_one_call_then_caches(repo_root: Path) -> None:
    url = "https://example.gov/fresh.pdf"
    body = b"fresh bytes"
    route = respx.get(url).mock(return_value=httpx.Response(200, content=body))

    service = IngestService(
        repo_root=repo_root,
        config=ConfigRepository(repo_root=repo_root),
    )
    entry = _entry_url(url, sha=compute_sha256(body))

    meta1, raw1 = service.load_document_bytes(entry)
    assert route.call_count == 1
    assert raw1 == body
    # Second call: cache short-circuits the network because sha is pinned.
    meta2, raw2 = service.load_document_bytes(entry)
    assert route.call_count == 1, "second call must hit cache, not network"
    assert meta1.sha256 == meta2.sha256 == compute_sha256(body)


@respx.mock
def test_http_retries_on_transient_503(repo_root: Path) -> None:
    url = "https://example.gov/flaky.pdf"
    body = b"ok at last"
    # First two calls 503, third succeeds.
    route = respx.get(url).mock(
        side_effect=[
            httpx.Response(503),
            httpx.Response(503),
            httpx.Response(200, content=body),
        ]
    )

    service = IngestService(
        repo_root=repo_root,
        config=ConfigRepository(repo_root=repo_root),
        max_http_attempts=3,
    )
    entry = _entry_url(url)
    _, raw = service.load_document_bytes(entry)
    assert raw == body
    assert route.call_count == 3, "expected 2 retries + 1 success"


@respx.mock
def test_http_raises_on_4xx_no_retry(repo_root: Path) -> None:
    url = "https://example.gov/forbidden.pdf"
    route = respx.get(url).mock(return_value=httpx.Response(403))

    service = IngestService(
        repo_root=repo_root,
        config=ConfigRepository(repo_root=repo_root),
        max_http_attempts=3,
    )
    entry = _entry_url(url)
    with pytest.raises(IngestError):
        service.load_document_bytes(entry)
    assert route.call_count == 1, "4xx must NOT retry"


def test_entry_without_url_or_local_raises(repo_root: Path, service: IngestService) -> None:
    entry = SourceRegistryEntry(
        source_id="bad",
        jurisdiction="SAMPLE",
        title="bad",
        source_url=None,
        document_type=DocumentType.STATUTE,
        authority_tier=AuthorityTier.TIER_1_STATUTE,
        local_path=None,
    )
    with pytest.raises(IngestError, match="local_path or source_url"):
        service.load_document_bytes(entry)


# ─── provenance helpers ─────────────────────────────────────────────────────


def test_compute_sha256_deterministic() -> None:
    assert compute_sha256(b"abc") == compute_sha256(b"abc")
    assert len(compute_sha256(b"abc")) == 64
    assert compute_sha256(b"abc") != compute_sha256(b"abd")
