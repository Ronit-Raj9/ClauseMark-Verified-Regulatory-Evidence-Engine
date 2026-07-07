"""Unit tests for the VLM-OCR extractor + the §11 graceful-degradation stub."""

from __future__ import annotations

import io
import logging

import pymupdf  # type: ignore[import-untyped]
import pytest
from rie_contracts import DocumentMeta, ElementType, OcrEngine
from rie_extract.adapters.vlm_ocr import (
    ScannedNotEnabledExtractor,
    VlmOcrExtractor,
    _ocr_confidence_threshold,
)
from rie_extract.vlm_client import FakeVlmClient

# ---------------------------------------------------------------------------
# Test fixtures: produce a one-page PDF on the fly so we don't depend on a
# pre-baked scanned binary in the repo. The PDF text content is irrelevant
# because the fake VLM client ignores pixels.
# ---------------------------------------------------------------------------


def _one_page_pdf(text: str = "placeholder") -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def _two_page_pdf() -> bytes:
    doc = pymupdf.open()
    for label in ("page-1", "page-2"):
        page = doc.new_page()
        page.insert_text((72, 72), label)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


# ---------------------------------------------------------------------------
# VlmOcrExtractor
# ---------------------------------------------------------------------------


def test_vlm_extractor_produces_vlm_tagged_element(dpa_meta: DocumentMeta) -> None:
    raw = _one_page_pdf()
    extractor = VlmOcrExtractor(vlm_client=FakeVlmClient("Section 1. Hello."))
    elements, _edges = extractor.extract(dpa_meta, raw)

    assert elements, "expected at least one element from VLM extraction"
    assert any("Hello" in el.text for el in elements), (
        f"expected element text containing 'Hello'; got {[el.text for el in elements]}"
    )
    assert all(el.ocr_engine == OcrEngine.VLM for el in elements)
    assert all(el.corrected is False for el in elements)


def test_vlm_extractor_renders_one_call_per_page(dpa_meta: DocumentMeta) -> None:
    raw = _two_page_pdf()
    client = FakeVlmClient(canned_text=["First page text.", "Second page text."])
    extractor = VlmOcrExtractor(vlm_client=client)
    extractor.extract(dpa_meta, raw)

    assert client.calls == 2, f"expected 2 VLM calls (one per page); got {client.calls}"


def test_vlm_extractor_parses_section_heading(dpa_meta: DocumentMeta) -> None:
    raw = _one_page_pdf()
    extractor = VlmOcrExtractor(
        vlm_client=FakeVlmClient(
            "Section 26. Transfer of personal data.\n\nAn organisation shall not transfer."
        )
    )
    elements, _edges = extractor.extract(dpa_meta, raw)
    sections = [
        el
        for el in elements
        if el.element_type == ElementType.SECTION and el.legal_numbering == "s. 26"
    ]
    assert sections, (
        f"expected a Section 26 element with canonical numbering; "
        f"got types {[(el.element_type, el.legal_numbering) for el in elements]}"
    )


def test_vlm_low_confidence_warns_but_keeps_element(
    dpa_meta: DocumentMeta, caplog: pytest.LogCaptureFixture
) -> None:
    raw = _one_page_pdf()
    client = FakeVlmClient("Some text.", confidence=0.3)  # below default 0.7
    extractor = VlmOcrExtractor(vlm_client=client)

    with caplog.at_level(logging.WARNING, logger="rie_extract.adapters.vlm_ocr"):
        elements, _edges = extractor.extract(dpa_meta, raw)

    assert elements, "low confidence must not drop the element"
    assert any("below confidence threshold" in rec.message for rec in caplog.records)
    assert all(el.corrected is False for el in elements)
    # extraction_confidence must reflect the (low) value, not be silently lifted.
    assert all(el.extraction_confidence == pytest.approx(0.3) for el in elements)


def test_vlm_path_logs_provenance_at_info(
    dpa_meta: DocumentMeta, caplog: pytest.LogCaptureFixture
) -> None:
    raw = _one_page_pdf()
    extractor = VlmOcrExtractor(vlm_client=FakeVlmClient("Section 1. Hello."))

    with caplog.at_level(logging.INFO, logger="rie_extract.adapters.vlm_ocr"):
        extractor.extract(dpa_meta, raw)

    assert any("vlm-ocr path engaged" in rec.message for rec in caplog.records), (
        "must log VLM provenance at INFO so the audit trail captures the path"
    )


def test_vlm_empty_transcription_still_emits_element(dpa_meta: DocumentMeta) -> None:
    """§11 graceful degradation: empty page is recorded, not dropped silently."""
    raw = _one_page_pdf()
    extractor = VlmOcrExtractor(vlm_client=FakeVlmClient("", confidence=0.0))
    elements, _edges = extractor.extract(dpa_meta, raw)
    assert len(elements) == 1
    assert elements[0].text == ""
    assert elements[0].ocr_engine == OcrEngine.VLM
    assert elements[0].extraction_confidence == 0.0


def test_vlm_runs_structure_graph_over_synthesised_elements(dpa_meta: DocumentMeta) -> None:
    raw = _one_page_pdf()
    extractor = VlmOcrExtractor(
        vlm_client=FakeVlmClient(
            "Section 5. Application.\n\nThis Act applies as set out in section 6.\n\n"
            "Section 6. Scope.\n\nThis Act applies to all controllers."
        )
    )
    _elements, edges = extractor.extract(dpa_meta, raw)
    # The cross-reference from s.5 → s.6 should be picked up by the
    # deterministic structure-graph builder run by the adapter.
    assert any(e.edge_type.value == "cross_reference" for e in edges), (
        f"expected a cross_reference edge; got {[e.edge_type.value for e in edges]}"
    )


# ---------------------------------------------------------------------------
# OCR_CONFIDENCE_THRESHOLD env override
# ---------------------------------------------------------------------------


def test_confidence_threshold_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OCR_CONFIDENCE_THRESHOLD", "0.9")
    assert _ocr_confidence_threshold() == pytest.approx(0.9)


def test_confidence_threshold_env_invalid_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OCR_CONFIDENCE_THRESHOLD", "not-a-number")
    assert _ocr_confidence_threshold() == pytest.approx(0.7)


def test_confidence_threshold_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OCR_CONFIDENCE_THRESHOLD", raising=False)
    assert _ocr_confidence_threshold() == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# ScannedNotEnabledExtractor — §11 graceful degradation
# ---------------------------------------------------------------------------


def test_scanned_not_enabled_returns_empty_placeholder(
    dpa_meta: DocumentMeta, caplog: pytest.LogCaptureFixture
) -> None:
    extractor = ScannedNotEnabledExtractor()
    with caplog.at_level(logging.WARNING, logger="rie_extract.adapters.vlm_ocr"):
        elements, edges = extractor.extract(dpa_meta, b"")
    assert len(elements) == 1
    assert elements[0].text == ""
    assert elements[0].extraction_confidence == 0.0
    assert elements[0].corrected is False
    assert edges == []
    assert any("VLM-OCR is disabled" in rec.message for rec in caplog.records)


def test_scanned_not_enabled_never_crashes_on_unreadable_bytes(
    dpa_meta: DocumentMeta,
) -> None:
    """The point of the stub: garbage bytes do NOT propagate as a crash."""
    extractor = ScannedNotEnabledExtractor()
    elements, _edges = extractor.extract(dpa_meta, b"\x00\x01not a pdf")
    assert elements and elements[0].extraction_confidence == 0.0
