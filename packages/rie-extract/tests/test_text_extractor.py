"""End-to-end test: plain-text extractor over the sample DPA."""

from __future__ import annotations

from pathlib import Path

from rie_contracts import DocumentMeta, ElementType
from rie_extract.adapters.text_extractor import TextExtractor


def test_extracts_minimum_paragraph_count(sample_dpa_bytes: bytes, dpa_meta: DocumentMeta) -> None:
    elements, _edges = TextExtractor().extract(dpa_meta, sample_dpa_bytes)
    paragraphs = [e for e in elements if e.element_type == ElementType.PARAGRAPH]
    assert len(paragraphs) >= 10, (
        f"expected ≥10 paragraph elements, got {len(paragraphs)}; total elements = {len(elements)}"
    )


def test_section_26_present_with_numbering(sample_dpa_bytes: bytes, dpa_meta: DocumentMeta) -> None:
    elements, _edges = TextExtractor().extract(dpa_meta, sample_dpa_bytes)
    s26 = [
        e
        for e in elements
        if e.element_type == ElementType.SECTION and e.legal_numbering and "26" in e.legal_numbering
    ]
    assert s26, "section 26 element missing"
    # Its paragraph children should mention the operative phrase.
    children = [e for e in elements if e.parent_id == s26[0].element_id]
    joined = " ".join(c.text for c in children)
    assert "transfer any personal data" in joined.lower()


def test_section_26_text_contains_operative_phrase(
    sample_dpa_bytes: bytes, dpa_meta: DocumentMeta
) -> None:
    elements, _edges = TextExtractor().extract(dpa_meta, sample_dpa_bytes)
    full = " ".join(e.text for e in elements).lower()
    assert "transfer any personal data outside the country" in full


def test_char_offsets_are_monotone(sample_dpa_bytes: bytes, dpa_meta: DocumentMeta) -> None:
    elements, _edges = TextExtractor().extract(dpa_meta, sample_dpa_bytes)
    last_start = -1
    for el in elements:
        assert el.char_start >= last_start, (
            f"non-monotone char_start at {el.element_id}: {el.char_start} < {last_start}"
        )
        last_start = el.char_start
        assert el.char_end >= el.char_start


def test_offsets_are_within_document(
    sample_dpa_path: Path, sample_dpa_bytes: bytes, dpa_meta: DocumentMeta
) -> None:
    text = sample_dpa_path.read_text(encoding="utf-8")
    elements, _edges = TextExtractor().extract(dpa_meta, sample_dpa_bytes)
    for el in elements:
        assert el.char_end <= len(text), (
            f"{el.element_id} char_end {el.char_end} exceeds document length {len(text)}"
        )
