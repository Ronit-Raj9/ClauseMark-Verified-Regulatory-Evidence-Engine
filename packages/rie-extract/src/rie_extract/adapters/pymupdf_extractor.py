"""PyMuPDF-backed extractor for born-digital PDFs.

Walks each page in reading order, emits one element per text block,
captures the block's bounding box and page, computes char offsets
against a single concatenated extracted-text buffer (the same buffer
that downstream verification gates will re-read).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import pymupdf  # type: ignore[import-untyped]
from rie_contracts import (
    BoundingBox,
    DocumentMeta,
    Element,
    ElementType,
    OcrEngine,
    StructureEdge,
)

from rie_extract.errors import ExtractionError
from rie_extract.structure.legal_numbering import parse_numbering

_HEADING_RE = re.compile(
    r"^\s*(?:CHAPTER\s+(?:[IVXLCDM]+|\d+)|"
    r"(?:Section|Sec\.?|s\.)\s+\d+[A-Za-z]?|"
    r"(?:Article|Art\.?)\s+\d+[A-Za-z]?)\b",
    re.IGNORECASE,
)


@dataclass
class PyMuPdfExtractor:
    """Born-digital PDF extractor."""

    def extract(
        self, doc_meta: DocumentMeta, raw: bytes
    ) -> tuple[Sequence[Element], Sequence[StructureEdge]]:
        try:
            pdf = pymupdf.open(stream=raw, filetype="pdf")
        except Exception as exc:  # pragma: no cover — surfacing pymupdf errors
            raise ExtractionError(f"PyMuPDF failed to open document: {exc}") from exc

        elements: list[Element] = []
        running_offset = 0
        chapter_id: str | None = None
        section_id: str | None = None
        index = 0

        try:
            for page_index, page in enumerate(pdf, start=1):
                blocks = page.get_text("blocks")  # (x0,y0,x1,y1, text, block_no, block_type)
                # Sort blocks in reading order (top-to-bottom, left-to-right).
                blocks.sort(key=lambda b: (round(b[1], 1), round(b[0], 1)))
                for block in blocks:
                    x0, y0, x1, y1, text, *_rest = block
                    if not isinstance(text, str):
                        continue
                    stripped = text.strip()
                    if not stripped:
                        # Still advance the offset to mirror full-text concatenation.
                        running_offset += len(text)
                        continue

                    bbox = BoundingBox(
                        page=page_index, x0=float(x0), y0=float(y0), x1=float(x1), y1=float(y1)
                    )
                    is_heading = bool(_HEADING_RE.match(stripped))
                    first_line = stripped.splitlines()[0]
                    numbering = parse_numbering(first_line) if is_heading else None

                    if is_heading:
                        head_lower = first_line.lower()
                        if "chapter" in head_lower:
                            etype = ElementType.HEADING
                        elif head_lower.startswith(("article", "art.")):
                            etype = ElementType.ARTICLE
                        else:
                            etype = ElementType.SECTION
                    else:
                        etype = ElementType.PARAGRAPH

                    char_start = running_offset
                    char_end = char_start + len(text)

                    eid = (
                        f"{doc_meta.doc_id}_{numbering.replace(' ', '').replace('.', '')}"
                        if numbering
                        else f"{doc_meta.doc_id}_{etype.value}_{index}"
                    )
                    index += 1

                    parent_id: str | None
                    if etype == ElementType.HEADING:
                        parent_id = None
                        chapter_id = eid
                        section_id = None
                    elif etype in {ElementType.SECTION, ElementType.ARTICLE}:
                        parent_id = chapter_id
                        section_id = eid
                    else:
                        parent_id = section_id or chapter_id

                    elements.append(
                        Element(
                            element_id=eid,
                            doc_id=doc_meta.doc_id,
                            parent_id=parent_id,
                            element_type=etype,
                            text=re.sub(r"\s+", " ", stripped),
                            page=page_index,
                            bbox=bbox,
                            char_start=char_start,
                            char_end=char_end,
                            extraction_confidence=0.95,
                            ocr_engine=OcrEngine.NONE,
                            legal_numbering=numbering,
                        )
                    )

                    running_offset = char_end
        finally:
            pdf.close()

        return elements, []
