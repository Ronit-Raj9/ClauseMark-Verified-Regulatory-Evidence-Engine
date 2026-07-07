"""Tests for env-toggled VLM routing in :mod:`rie_extract.router`."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import pytest
from rie_contracts import (
    AuthorityTier,
    DocumentMeta,
    DocumentType,
)
from rie_extract.adapters.born_digital_pdf_extractor import BornDigitalPdfExtractor
from rie_extract.adapters.vlm_ocr import (
    ScannedNotEnabledExtractor,
    VlmOcrExtractor,
)
from rie_extract.router import pick_adapter


def _scanned_meta() -> DocumentMeta:
    return DocumentMeta(
        doc_id="scanned_doc_2026",
        jurisdiction="SAMPLE",
        title="Scanned Privacy Regulation 2026",
        document_type=DocumentType.STATUTE,
        authority_tier=AuthorityTier.TIER_1_STATUTE,
        source_url="file:///data/samples/scanned_regulation.pdf",
        sha256="b" * 64,
        retrieved_at=datetime(2026, 5, 24, tzinfo=UTC),
        language="en",
    )


def test_router_returns_stub_when_vlm_disabled(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delenv("OLLAMA_VLM_MODEL", raising=False)
    with caplog.at_level(logging.WARNING, logger="rie_extract.router"):
        adapter = pick_adapter(_scanned_meta())
    assert isinstance(adapter, ScannedNotEnabledExtractor)
    assert any("graceful degradation" in rec.message for rec in caplog.records)


def test_router_returns_vlm_when_env_var_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_VLM_MODEL", "llama3.2-vision")
    adapter = pick_adapter(_scanned_meta())
    assert isinstance(adapter, VlmOcrExtractor)


def test_router_returns_vlm_when_explicit_flag_true(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OLLAMA_VLM_MODEL", raising=False)
    adapter = pick_adapter(_scanned_meta(), enable_vlm=True)
    assert isinstance(adapter, VlmOcrExtractor)


def test_router_explicit_false_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_VLM_MODEL", "llama3.2-vision")
    adapter = pick_adapter(_scanned_meta(), enable_vlm=False)
    assert isinstance(adapter, ScannedNotEnabledExtractor)


def test_non_scanned_routes_unaffected_by_vlm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setting the VLM env must NOT poison the .pdf / .html / .txt routes."""
    monkeypatch.setenv("OLLAMA_VLM_MODEL", "llama3.2-vision")
    meta = DocumentMeta(
        doc_id="born_digital_pdf",
        jurisdiction="SAMPLE",
        title="Some Statute",
        document_type=DocumentType.STATUTE,
        authority_tier=AuthorityTier.TIER_1_STATUTE,
        source_url="file:///data/samples/statute.pdf",
        sha256="c" * 64,
        retrieved_at=datetime(2026, 5, 24, tzinfo=UTC),
        language="en",
    )
    adapter = pick_adapter(meta)
    assert isinstance(adapter, BornDigitalPdfExtractor)
