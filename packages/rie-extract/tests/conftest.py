"""Shared fixtures for rie-extract unit tests."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from rie_contracts import AuthorityTier, DocumentMeta, DocumentType

# Project root: …/Digital_trade/packages/rie-extract/tests/conftest.py
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"


@pytest.fixture
def sample_dpa_path() -> Path:
    p = SAMPLES_DIR / "sample_dpa.txt"
    assert p.exists(), f"sample_dpa.txt missing at {p}"
    return p


@pytest.fixture
def sample_dpa_bytes(sample_dpa_path: Path) -> bytes:
    return sample_dpa_path.read_bytes()


@pytest.fixture
def dpa_meta() -> DocumentMeta:
    return DocumentMeta(
        doc_id="sample_dpa_2020",
        jurisdiction="SAMPLE",
        title="Sample Personal Data Protection Act, 2020",
        document_type=DocumentType.STATUTE,
        authority_tier=AuthorityTier.TIER_1_STATUTE,
        source_url="file:///data/samples/sample_dpa.txt",
        sha256="a" * 64,
        retrieved_at=datetime(2026, 5, 24, tzinfo=UTC),
        language="en",
    )
