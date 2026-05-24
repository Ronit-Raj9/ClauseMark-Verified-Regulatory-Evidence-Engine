"""Optional Docling-backed extractor — degrades gracefully if Docling absent.

Docling is a heavy dependency (model weights, native libs). The
extraction pipeline must not crash if the user's environment cannot
import it; instead, callers should fall back to ``PyMuPdfExtractor``
and a warning is logged.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass

from rie_contracts import (
    DocumentMeta,
    Element,
    ElementType,
    OcrEngine,
    StructureEdge,
)

from rie_extract.errors import ExtractionError
from rie_extract.structure.legal_numbering import parse_numbering

_LOG = logging.getLogger(__name__)


def docling_available() -> bool:
    """Return True iff a usable Docling install is importable."""
    try:
        from docling.document_converter import DocumentConverter  # noqa: F401
    except Exception as exc:  # pragma: no cover — depends on env
        _LOG.warning("Docling unavailable: %s", exc)
        return False
    return True


@dataclass
class DoclingExtractor:
    """Richer extraction with layout / reading-order awareness.

    Falls back to ``ExtractionError`` if docling cannot be loaded — the
    service layer is responsible for choosing a different adapter.
    """

    def extract(
        self, doc_meta: DocumentMeta, raw: bytes
    ) -> tuple[Sequence[Element], Sequence[StructureEdge]]:
        try:
            from docling.document_converter import (
                DocumentConverter,  # type: ignore[import-not-found]
            )
        except Exception as exc:
            _LOG.warning("Docling import failed (%s); caller should fall back.", exc)
            raise ExtractionError("docling is not installed or failed to import") from exc

        try:
            import io

            converter = DocumentConverter()
            # Newer Docling APIs accept a stream / Path; we use a tempfile-ish
            # adaptation that keeps memory in-process.
            result = converter.convert(io.BytesIO(raw))
            doc = getattr(result, "document", result)
        except Exception as exc:
            _LOG.warning("Docling conversion failed: %s", exc)
            raise ExtractionError(f"docling conversion failed: {exc}") from exc

        elements: list[Element] = []
        running_offset = 0
        index = 0
        current_parent: str | None = None

        # Best-effort traversal — Docling versions vary in their object
        # shape; we read text via duck-typed accessors and emit a flat
        # element stream. Service-level structure graph fills in the rest.
        items = getattr(doc, "iterate_items", None)
        iterable = items() if callable(items) else getattr(doc, "items", [])
        for item in iterable:
            text = getattr(item, "text", None) or ""
            if not text.strip():
                continue
            label = (getattr(item, "label", "") or "").lower()
            page = int(getattr(item, "page", 1) or 1)

            if label in {"title", "section_header", "heading"}:
                numbering = parse_numbering(text)
                if numbering and numbering.startswith("s."):
                    etype = ElementType.SECTION
                elif numbering and numbering.startswith("art."):
                    etype = ElementType.ARTICLE
                else:
                    etype = ElementType.HEADING
            elif label in {"list_item"}:
                etype = ElementType.LIST_ITEM
                numbering = None
            elif label in {"table"}:
                etype = ElementType.TABLE
                numbering = None
            else:
                etype = ElementType.PARAGRAPH
                numbering = None

            normalised = re.sub(r"\s+", " ", text).strip()
            char_start = running_offset
            char_end = char_start + len(normalised)
            running_offset = char_end + 1

            eid = (
                f"{doc_meta.doc_id}_{numbering.replace(' ', '').replace('.', '')}"
                if numbering
                else f"{doc_meta.doc_id}_{etype.value}_{index}"
            )
            index += 1

            if etype in {ElementType.SECTION, ElementType.ARTICLE, ElementType.HEADING}:
                current_parent = eid
                parent_id = None
            else:
                parent_id = current_parent

            elements.append(
                Element(
                    element_id=eid,
                    doc_id=doc_meta.doc_id,
                    parent_id=parent_id,
                    element_type=etype,
                    text=normalised,
                    page=page,
                    bbox=None,
                    char_start=char_start,
                    char_end=char_end,
                    extraction_confidence=0.9,
                    ocr_engine=OcrEngine.NONE,
                    legal_numbering=numbering,
                )
            )

        return elements, []
