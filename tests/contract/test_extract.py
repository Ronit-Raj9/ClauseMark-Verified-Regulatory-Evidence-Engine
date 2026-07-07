"""Contract test for `DocumentExtractorPort` against the rie-extract adapter.

This test passes with any conforming implementation. It exercises:
  • runtime `isinstance` against the Protocol
  • end-to-end extraction on the real sample DPA text
  • presence of the operative cross-border-transfer phrase in a real element
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from rie_contracts import (
    AuthorityTier,
    DocumentExtractorPort,
    DocumentMeta,
    DocumentType,
    Element,
    ElementType,
    StructureEdge,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DPA = PROJECT_ROOT / "data" / "samples" / "sample_dpa.txt"


@pytest.fixture
def extractor() -> DocumentExtractorPort:
    """Bind to the rie-extract adapter through the public entrypoint."""
    from rie_extract import ExtractionService

    return ExtractionService()


@pytest.fixture
def dpa_meta() -> DocumentMeta:
    return DocumentMeta(
        doc_id="sample_dpa_2020",
        jurisdiction="SAMPLE",
        title="Sample Personal Data Protection Act, 2020",
        document_type=DocumentType.STATUTE,
        authority_tier=AuthorityTier.TIER_1_STATUTE,
        source_url="file://data/samples/sample_dpa.txt",
        sha256="a" * 64,
        retrieved_at=datetime(2026, 5, 24, tzinfo=UTC),
        language="en",
    )


def test_extractor_satisfies_protocol(extractor: DocumentExtractorPort) -> None:
    assert isinstance(extractor, DocumentExtractorPort)


def test_extract_returns_elements_and_edges(
    extractor: DocumentExtractorPort, dpa_meta: DocumentMeta
) -> None:
    raw = SAMPLE_DPA.read_bytes()
    elements, edges = extractor.extract(dpa_meta, raw)
    assert isinstance(elements, (list, tuple))
    assert isinstance(edges, (list, tuple))
    assert all(isinstance(e, Element) for e in elements)
    assert all(isinstance(e, StructureEdge) for e in edges)
    assert len(elements) >= 10, f"expected ≥10 elements; got {len(elements)}"


def test_extract_finds_cross_border_transfer_clause(
    extractor: DocumentExtractorPort, dpa_meta: DocumentMeta
) -> None:
    raw = SAMPLE_DPA.read_bytes()
    elements, _edges = extractor.extract(dpa_meta, raw)
    matches = [
        e for e in elements if "transfer any personal data outside the country" in e.text.lower()
    ]
    assert matches, (
        "No element contains the operative cross-border transfer phrase; "
        f"sample element texts: {[e.text[:80] for e in elements[:5]]}"
    )


def test_extract_preserves_section_26_numbering(
    extractor: DocumentExtractorPort, dpa_meta: DocumentMeta
) -> None:
    raw = SAMPLE_DPA.read_bytes()
    elements, _edges = extractor.extract(dpa_meta, raw)
    s26 = [
        e
        for e in elements
        if e.element_type == ElementType.SECTION and e.legal_numbering and "26" in e.legal_numbering
    ]
    assert s26, "Section 26 missing from extracted elements"


def test_extract_emits_structure_edges(
    extractor: DocumentExtractorPort, dpa_meta: DocumentMeta
) -> None:
    raw = SAMPLE_DPA.read_bytes()
    _elements, edges = extractor.extract(dpa_meta, raw)
    # The sample DPA has cross-references (s.2(c) → s. 30; s.5 → s. 6;
    # s.26(2) → s. 26(1)) so at least one edge must be present.
    assert edges, "expected at least one structure edge"


def test_scanned_path_without_vlm_degrades_gracefully(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """§11 graceful-degradation: scanned hint + VLM disabled must NOT crash.

    Instead, the router returns a stub adapter that emits a single empty
    element with confidence=0 so coverage routing sees the document as
    *attempted but unread*. This is the keystone behaviour for the
    "scanned-not-enabled" path and is part of the public contract.
    """
    monkeypatch.delenv("OLLAMA_VLM_MODEL", raising=False)
    from rie_extract.router import pick_adapter

    scanned_meta = DocumentMeta(
        doc_id="scanned_doc_2026",
        jurisdiction="SAMPLE",
        title="Scanned Privacy Regulation",
        document_type=DocumentType.STATUTE,
        authority_tier=AuthorityTier.TIER_1_STATUTE,
        source_url="file:///data/samples/scanned_regulation.pdf",
        sha256="d" * 64,
        retrieved_at=datetime(2026, 5, 24, tzinfo=UTC),
        language="en",
    )
    adapter = pick_adapter(scanned_meta)
    elements, edges = adapter.extract(scanned_meta, b"")

    assert len(elements) == 1
    assert elements[0].text == ""
    assert elements[0].extraction_confidence == 0.0
    assert elements[0].corrected is False
    assert edges == []


def test_born_digital_pdf_docling_chain_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    """§5.2: PyMuPDF text is merged with mocked Docling structure when available."""
    from rie_contracts import Element, ElementType, StructureEdge, StructureEdgeType
    from rie_extract.adapters.born_digital_pdf_extractor import BornDigitalPdfExtractor

    docling_elements = [
        Element(
            element_id="test_pdf_p1",
            doc_id="test_pdf",
            element_type=ElementType.SECTION,
            text="Section 1",
            page=1,
            char_start=0,
            char_end=9,
            extraction_confidence=0.95,
            legal_numbering="1",
        )
    ]
    docling_edges = [
        StructureEdge(
            from_element="test_pdf_p1",
            to_element="test_pdf_p2",
            edge_type=StructureEdgeType.CROSS_REFERENCE,
            raw_reference="s. 2",
        )
    ]

    class _MockDoclingExtractor:
        def extract(self, doc_meta, raw):
            del doc_meta, raw
            return docling_elements, docling_edges

    monkeypatch.setattr(
        "rie_extract.adapters.born_digital_pdf_extractor.docling_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "rie_extract.adapters.born_digital_pdf_extractor.DoclingExtractor",
        _MockDoclingExtractor,
    )

    raw = (
        b"%PDF-1.4\n1 0 obj<<>>endobj\n"
        b"2 0 obj<</Type/Catalog/Pages 3 0 R>>endobj\n"
        b"3 0 obj<</Type/Pages/Kids[4 0 R]/Count 1>>endobj\n"
        b"4 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 3 0 R>>endobj\n"
        b"xref\n0 5\n0000000000 65535 f \n"
        b"trailer<</Size 5/Root 2 0 R>>\nstartxref\n0\n%%EOF"
    )
    meta = DocumentMeta(
        doc_id="test_pdf",
        jurisdiction="SAMPLE",
        title="sample.pdf",
        document_type=DocumentType.STATUTE,
        authority_tier=AuthorityTier.TIER_1_STATUTE,
        sha256="b" * 64,
        retrieved_at=datetime(2026, 5, 24, tzinfo=UTC),
        language="en",
    )
    elements, edges = BornDigitalPdfExtractor().extract(meta, raw)

    assert isinstance(elements, list)
    assert edges == docling_edges


def test_vlm_extractor_marks_elements_with_vlm_engine_and_not_corrected() -> None:
    """Per CLAUDE.md / §5.2: VLM output is NEVER labelled 'faithful'.

    Every VLM-derived element must carry ``ocr_engine == VLM`` and
    ``corrected is False`` until a human edits it.
    """
    import io

    import pymupdf  # type: ignore[import-untyped]
    from rie_contracts import OcrEngine
    from rie_extract.adapters.vlm_ocr import VlmOcrExtractor
    from rie_extract.vlm_client import FakeVlmClient

    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "placeholder")
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()

    meta = DocumentMeta(
        doc_id="scanned_doc_2026",
        jurisdiction="SAMPLE",
        title="Scanned Privacy Regulation",
        document_type=DocumentType.STATUTE,
        authority_tier=AuthorityTier.TIER_1_STATUTE,
        sha256="e" * 64,
        retrieved_at=datetime(2026, 5, 24, tzinfo=UTC),
        language="en",
    )
    extractor = VlmOcrExtractor(vlm_client=FakeVlmClient("Section 1. Hello world."))
    elements, _edges = extractor.extract(meta, buf.getvalue())

    assert elements
    assert all(el.ocr_engine == OcrEngine.VLM for el in elements)
    assert all(el.corrected is False for el in elements)
