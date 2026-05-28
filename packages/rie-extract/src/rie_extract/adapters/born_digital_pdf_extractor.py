"""Born-digital PDF chain: PyMuPDF (text) then Docling (structure).

Per systemArchitecture §5.2: text is taken from the PDF text layer via
PyMuPDF; Docling recovers layout, reading order, and tables when available.
If Docling is absent or fails, PyMuPDF output alone is returned.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from rie_contracts import DocumentMeta, Element, ElementType, StructureEdge

from rie_extract.adapters.docling_extractor import DoclingExtractor, docling_available
from rie_extract.adapters.pymupdf_extractor import PyMuPdfExtractor
from rie_extract.errors import ExtractionError

_LOG = logging.getLogger(__name__)


@dataclass
class BornDigitalPdfExtractor:
    """PyMuPDF → Docling chain for born-digital PDFs."""

    def extract(
        self, doc_meta: DocumentMeta, raw: bytes
    ) -> tuple[Sequence[Element], Sequence[StructureEdge]]:
        pymupdf_elements, _ = PyMuPdfExtractor().extract(doc_meta, raw)
        if not docling_available():
            _LOG.info(
                "BornDigitalPdfExtractor: Docling unavailable for doc_id=%s — PyMuPDF only",
                doc_meta.doc_id,
            )
            return pymupdf_elements, []

        try:
            docling_elements, docling_edges = DoclingExtractor().extract(doc_meta, raw)
        except ExtractionError as exc:
            _LOG.warning(
                "BornDigitalPdfExtractor: Docling failed for doc_id=%s (%s) — PyMuPDF only",
                doc_meta.doc_id,
                exc,
            )
            return pymupdf_elements, []

        merged = _merge_pymupdf_text_docling_structure(
            list(pymupdf_elements), list(docling_elements)
        )
        _LOG.info(
            "BornDigitalPdfExtractor: merged PyMuPDF text + Docling structure "
            "(%d elements, %d edges) for doc_id=%s",
            len(merged),
            len(docling_edges),
            doc_meta.doc_id,
        )
        return merged, docling_edges


def _merge_pymupdf_text_docling_structure(
    pymupdf_elements: list[Element],
    docling_elements: list[Element],
) -> list[Element]:
    """Keep PyMuPDF text/offsets; enrich element types and parent hierarchy from Docling."""
    if not docling_elements:
        return pymupdf_elements

    docling_by_page: dict[int, list[Element]] = {}
    for element in docling_elements:
        docling_by_page.setdefault(element.page, []).append(element)

    merged: list[Element] = []
    for pm_element in pymupdf_elements:
        page_candidates = docling_by_page.get(pm_element.page, [])
        docling_match = _best_docling_match(pm_element, page_candidates)
        if docling_match is None:
            merged.append(pm_element)
            continue

        updates: dict[str, object] = {}
        if docling_match.element_type != ElementType.PARAGRAPH:
            updates["element_type"] = docling_match.element_type
        if docling_match.parent_id is not None:
            updates["parent_id"] = docling_match.parent_id
        if docling_match.legal_numbering and not pm_element.legal_numbering:
            updates["legal_numbering"] = docling_match.legal_numbering

        merged.append(pm_element.model_copy(update=updates) if updates else pm_element)
    return merged


def _best_docling_match(pymupdf_element: Element, candidates: list[Element]) -> Element | None:
    """Pick the Docling element whose normalised text best overlaps PyMuPDF text."""
    if not candidates:
        return None
    pm_text = _normalise(pymupdf_element.text)
    if not pm_text:
        return None

    best: Element | None = None
    best_score = 0.0
    for candidate in candidates:
        dl_text = _normalise(candidate.text)
        if not dl_text:
            continue
        if pm_text in dl_text or dl_text in pm_text:
            score = min(len(pm_text), len(dl_text)) / max(len(pm_text), len(dl_text))
        else:
            pm_tokens = set(pm_text.split())
            dl_tokens = set(dl_text.split())
            if not pm_tokens or not dl_tokens:
                continue
            score = len(pm_tokens & dl_tokens) / len(pm_tokens | dl_tokens)
        if score > best_score:
            best_score = score
            best = candidate

    return best if best_score >= 0.3 else None


def _normalise(text: str) -> str:
    return " ".join(text.split()).lower()
