"""Tests for born-digital PDF routing (PyMuPDF → Docling chain)."""

from __future__ import annotations

import io
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pymupdf  # type: ignore[import-untyped]
import pytest
from rie_contracts import (
    AuthorityTier,
    DocumentMeta,
    DocumentType,
    Element,
    ElementType,
    OcrEngine,
)

from rie_extract.adapters.born_digital_pdf_extractor import BornDigitalPdfExtractor
from rie_extract.router import pick_adapter
from rie_extract.service import ExtractionService


def _pdf_meta() -> DocumentMeta:
    return DocumentMeta(
        doc_id="test_pdf",
        jurisdiction="SAMPLE",
        title="sample.pdf",
        document_type=DocumentType.STATUTE,
        effective_date=None,
        authority_tier=AuthorityTier.TIER_1_STATUTE,
        source_url=None,
        sha256="a" * 64,
        retrieved_at=datetime.now(tz=UTC),
        language="en",
    )


def _one_page_pdf(text: str = "Section 5. Application.\n\nSubject to section 6.") -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def test_router_picks_born_digital_chain_for_pdf() -> None:
    adapter = pick_adapter(_pdf_meta(), source_path="/tmp/sample.pdf")
    assert isinstance(adapter, BornDigitalPdfExtractor)


def test_born_digital_extractor_handles_minimal_pdf_bytes() -> None:
    """Minimal valid PDF — PyMuPDF path must not crash even if Docling absent."""
    raw = (
        b"%PDF-1.4\n1 0 obj<<>>endobj\n"
        b"2 0 obj<</Type/Catalog/Pages 3 0 R>>endobj\n"
        b"3 0 obj<</Type/Pages/Kids[4 0 R]/Count 1>>endobj\n"
        b"4 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 3 0 R>>endobj\n"
        b"xref\n0 5\n0000000000 65535 f \n"
        b"trailer<</Size 5/Root 2 0 R>>\nstartxref\n0\n%%EOF"
    )
    elements, edges = BornDigitalPdfExtractor().extract(_pdf_meta(), raw)
    assert isinstance(elements, list)
    assert isinstance(edges, list)


def test_born_digital_chain_invokes_docling_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """§5.2: PyMuPDF runs first; Docling enriches when importable."""
    meta = _pdf_meta()
    raw = _one_page_pdf()

    docling_element = Element(
        element_id="test_pdf_section_5",
        doc_id=meta.doc_id,
        parent_id=None,
        element_type=ElementType.SECTION,
        text="Section 5. Application.",
        page=1,
        bbox=None,
        char_start=0,
        char_end=24,
        extraction_confidence=0.9,
        ocr_engine=OcrEngine.NONE,
        legal_numbering="s. 5",
    )
    mock_docling = MagicMock(
        return_value=([docling_element], []),
    )

    monkeypatch.setattr(
        "rie_extract.adapters.born_digital_pdf_extractor.docling_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "rie_extract.adapters.born_digital_pdf_extractor.DoclingExtractor",
        lambda: MagicMock(extract=mock_docling),
    )

    elements, edges = BornDigitalPdfExtractor().extract(meta, raw)

    mock_docling.assert_called_once()
    assert elements
    assert any(el.text for el in elements)
    assert isinstance(edges, list)


def test_born_digital_falls_back_to_pymupdf_when_docling_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "rie_extract.adapters.born_digital_pdf_extractor.docling_available",
        lambda: False,
    )
    raw = _one_page_pdf()
    elements, edges = BornDigitalPdfExtractor().extract(_pdf_meta(), raw)
    assert elements
    assert edges == []


def test_extraction_service_runs_structure_graph_for_pdf() -> None:
    """Structure graph (§5.3) must still run after the §5.2 extract chain."""
    meta = _pdf_meta()
    raw = _one_page_pdf(
        "Section 5. Application.\n\nThis Act applies as set out in section 6.\n\n"
        "Section 6. Scope.\n\nThis Act applies to all controllers."
    )
    elements, edges = ExtractionService().extract(meta, raw)
    assert elements
    assert any(e.edge_type.value == "cross_reference" for e in edges), (
        f"expected cross_reference edges from structure graph; got {[e.edge_type.value for e in edges]}"
    )
