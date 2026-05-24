"""BeautifulSoup-backed HTML extractor."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from bs4 import BeautifulSoup, Tag
from rie_contracts import (
    DocumentMeta,
    Element,
    ElementType,
    OcrEngine,
    StructureEdge,
)

from rie_extract.structure.legal_numbering import parse_numbering

_HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})


@dataclass
class HtmlExtractor:
    """Walks an HTML document and emits one element per semantic block."""

    def extract(
        self, doc_meta: DocumentMeta, raw: bytes
    ) -> tuple[Sequence[Element], Sequence[StructureEdge]]:
        soup = BeautifulSoup(raw, "lxml")
        # Strip noise.
        for noise in soup(["script", "style", "noscript"]):
            noise.decompose()

        body = soup.body or soup
        elements: list[Element] = []
        running_offset = 0
        index = 0
        current_section_id: str | None = None

        for tag in body.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "table"]):
            if not isinstance(tag, Tag):
                continue
            text = tag.get_text(" ", strip=True)
            if not text:
                continue

            name = tag.name.lower()
            if name in _HEADING_TAGS:
                numbering = parse_numbering(text)
                etype = (
                    ElementType.SECTION
                    if numbering and numbering.startswith("s.")
                    else (
                        ElementType.ARTICLE
                        if numbering and numbering.startswith("art.")
                        else ElementType.HEADING
                    )
                )
                eid = (
                    f"{doc_meta.doc_id}_{numbering.replace(' ', '').replace('.', '')}"
                    if numbering
                    else f"{doc_meta.doc_id}_{etype.value}_{index}"
                )
                parent_id: str | None = None
                if etype in {ElementType.SECTION, ElementType.ARTICLE}:
                    current_section_id = eid
                else:
                    current_section_id = eid
            elif name == "li":
                etype = ElementType.LIST_ITEM
                numbering = None
                eid = f"{doc_meta.doc_id}_{etype.value}_{index}"
                parent_id = current_section_id
            elif name == "table":
                etype = ElementType.TABLE
                numbering = None
                eid = f"{doc_meta.doc_id}_{etype.value}_{index}"
                parent_id = current_section_id
            else:
                etype = ElementType.PARAGRAPH
                numbering = None
                eid = f"{doc_meta.doc_id}_{etype.value}_{index}"
                parent_id = current_section_id

            index += 1
            normalised = re.sub(r"\s+", " ", text).strip()
            char_start = running_offset
            char_end = char_start + len(normalised)
            running_offset = char_end + 1  # trailing newline between blocks

            elements.append(
                Element(
                    element_id=eid,
                    doc_id=doc_meta.doc_id,
                    parent_id=parent_id,
                    element_type=etype,
                    text=normalised,
                    page=1,
                    bbox=None,
                    char_start=char_start,
                    char_end=char_end,
                    extraction_confidence=0.9,
                    ocr_engine=OcrEngine.NONE,
                    legal_numbering=numbering,
                )
            )

        return elements, []
