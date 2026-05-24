"""Plain-text extractor — the deterministic fallback path.

Splits a statute / regulation expressed as plain text into ordered
`Element` rows:

  • CHAPTER N — emitted as a HEADING element.
  • "Section N. Title" / "Article N. Title" — emitted as a SECTION /
    ARTICLE element. The body that follows (up to the next heading) is
    attached as PARAGRAPH children. Numbered sub-paragraphs ``(1) …``
    and lettered sub-paragraphs ``(a) …`` are each their own element.
  • A definitions section's children are tagged DEFINITION.

`char_start` / `char_end` are taken against the *original* document
text so offsets are byte-identical for the verification gates.
"""

from __future__ import annotations

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

from rie_extract.structure.legal_numbering import parse_numbering

# Lines that begin a numbered heading.
_HEADING_PATTERNS = [
    re.compile(r"^\s*CHAPTER\s+([IVXLCDM]+|\d+)\b.*$"),
    re.compile(r"^\s*(?:Section|Sec\.?|s\.)\s+(\d+[A-Za-z]?)\.?\s*.*$", re.IGNORECASE),
    re.compile(r"^\s*(?:Article|Art\.?)\s+(\d+[A-Za-z]?)\.?\s*.*$", re.IGNORECASE),
]

# Sub-paragraph openers (lead-in tokens at the start of a line / paragraph).
_NUMBERED_SUB = re.compile(r"^\s*\((\d+)\)\s+")
_LETTERED_SUB = re.compile(r"^\s*\(([a-z]+)\)\s+")

_DEFINITIONS_HEADING = re.compile(r"\b(definitions?|interpretation)\b", re.IGNORECASE)


@dataclass
class _LineBlock:
    """Internal scratch type — a maximal run of non-blank lines plus its char span."""

    text: str
    char_start: int
    char_end: int


@dataclass
class TextExtractor:
    """Plain-text statute splitter. Produces a flat-then-parented element list."""

    def extract(
        self, doc_meta: DocumentMeta, raw: bytes
    ) -> tuple[Sequence[Element], Sequence[StructureEdge]]:
        text = raw.decode("utf-8", errors="replace")
        blocks = _blocks(text)
        elements = self._blocks_to_elements(doc_meta, blocks)
        # Edges are built by the service, not the adapter — keep adapters thin.
        return elements, []

    # ── Internals ───────────────────────────────────────────────────────

    def _blocks_to_elements(
        self, doc_meta: DocumentMeta, blocks: list[_LineBlock]
    ) -> list[Element]:
        out: list[Element] = []
        index = 0  # global running counter, for fallback ids
        current_chapter_id: str | None = None
        current_section_id: str | None = None
        current_section_is_defs = False
        # Sub-counters per parent for paragraph indexing
        para_counters: dict[str, int] = {}

        def _next_para_id(parent_id: str | None) -> tuple[str, int]:
            nonlocal index
            key = parent_id or "__root__"
            n = para_counters.get(key, 0) + 1
            para_counters[key] = n
            idx = index
            index += 1
            return key, idx

        for block in blocks:
            text = block.text.strip("\n")
            if not text.strip():
                continue

            heading_kind = _classify_heading(text)

            if heading_kind == "chapter":
                idx = index
                index += 1
                eid = f"{doc_meta.doc_id}_chapter_{idx}"
                numbering = parse_numbering(text.splitlines()[0])
                out.append(
                    Element(
                        element_id=eid,
                        doc_id=doc_meta.doc_id,
                        parent_id=None,
                        element_type=ElementType.HEADING,
                        text=text,
                        page=1,
                        bbox=None,
                        char_start=block.char_start,
                        char_end=block.char_end,
                        extraction_confidence=1.0,
                        ocr_engine=OcrEngine.NONE,
                        legal_numbering=numbering,
                    )
                )
                current_chapter_id = eid
                current_section_id = None
                current_section_is_defs = False
                continue

            if heading_kind in {"section", "article"}:
                # The block typically contains: "Section 26. Title.\nBody…".
                first_line, *rest_lines = text.splitlines()
                numbering = parse_numbering(first_line)
                etype = ElementType.SECTION if heading_kind == "section" else ElementType.ARTICLE
                eid = (
                    f"{doc_meta.doc_id}_{numbering.replace(' ', '').replace('.', '')}"
                    if numbering
                    else f"{doc_meta.doc_id}_{etype.value}_{index}"
                )
                index += 1
                # Compute char span of just the heading line.
                heading_end = block.char_start + len(first_line)
                out.append(
                    Element(
                        element_id=eid,
                        doc_id=doc_meta.doc_id,
                        parent_id=current_chapter_id,
                        element_type=etype,
                        text=first_line.strip(),
                        page=1,
                        bbox=None,
                        char_start=block.char_start,
                        char_end=heading_end,
                        extraction_confidence=1.0,
                        ocr_engine=OcrEngine.NONE,
                        legal_numbering=numbering,
                    )
                )
                current_section_id = eid
                current_section_is_defs = bool(_DEFINITIONS_HEADING.search(first_line))

                # Body paragraphs that follow on subsequent lines of the same block.
                if rest_lines:
                    body_start = block.char_start + len(first_line) + 1  # +1 for the newline
                    body_text = "\n".join(rest_lines)
                    # Compute the actual char_start in original text by searching
                    # forward from heading_end — handles repeated whitespace.
                    self._emit_body_paragraphs(
                        doc_meta=doc_meta,
                        body_text=body_text,
                        body_char_start=body_start,
                        parent_id=current_section_id,
                        is_defs=current_section_is_defs,
                        out=out,
                        next_para_id=_next_para_id,
                    )
                continue

            # Free-floating paragraph block — attach to current section if any.
            parent_id = current_section_id or current_chapter_id
            self._emit_body_paragraphs(
                doc_meta=doc_meta,
                body_text=text,
                body_char_start=block.char_start,
                parent_id=parent_id,
                is_defs=current_section_is_defs,
                out=out,
                next_para_id=_next_para_id,
            )

        return out

    def _emit_body_paragraphs(
        self,
        doc_meta: DocumentMeta,
        body_text: str,
        body_char_start: int,
        parent_id: str | None,
        is_defs: bool,
        out: list[Element],
        next_para_id,
    ) -> None:
        """Split a body block into paragraph elements with correct offsets."""
        paragraphs = _split_into_paragraphs(body_text)
        cursor = body_char_start
        for para_text in paragraphs:
            if not para_text.strip():
                # Still advance the cursor across whitespace-only segments.
                cursor += len(para_text)
                continue
            # Find the leading-whitespace offset so char_start lands on the
            # first non-space character of the paragraph.
            leading_ws = len(para_text) - len(para_text.lstrip())
            start = cursor + leading_ws
            stripped = para_text.strip("\n")
            end = cursor + len(para_text.rstrip("\n"))
            # Normalise textual content (collapse the indent that the source
            # uses for wrap-around lines into single spaces).
            display_text = re.sub(r"\s+", " ", stripped).strip()

            etype = ElementType.PARAGRAPH
            numbering: str | None = None
            sub_match = _NUMBERED_SUB.match(stripped)
            letter_match = _LETTERED_SUB.match(stripped)
            if (is_defs and letter_match) or is_defs:
                etype = ElementType.DEFINITION
            elif sub_match or letter_match:
                etype = ElementType.LIST_ITEM

            # Build a derived numbering for sub-paragraphs (links into parent's section).
            parent_numbering = next(
                (e.legal_numbering for e in reversed(out) if e.element_id == parent_id), None
            )
            if parent_numbering and sub_match:
                numbering = f"{parent_numbering}({sub_match.group(1)})"
            elif parent_numbering and letter_match:
                # Letter-only items inherit the closest preceding numbered sibling.
                preceding_sub = self._closest_preceding_numbered(out, parent_id)
                base = preceding_sub or parent_numbering
                numbering = f"{base}({letter_match.group(1)})"

            _key, idx = next_para_id(parent_id)
            eid = f"{doc_meta.doc_id}_{etype.value}_{idx}"

            out.append(
                Element(
                    element_id=eid,
                    doc_id=doc_meta.doc_id,
                    parent_id=parent_id,
                    element_type=etype,
                    text=display_text,
                    page=1,
                    bbox=None,
                    char_start=start,
                    char_end=end,
                    extraction_confidence=1.0,
                    ocr_engine=OcrEngine.NONE,
                    legal_numbering=numbering,
                )
            )
            cursor += len(para_text)

    def _closest_preceding_numbered(
        self, emitted: list[Element], parent_id: str | None
    ) -> str | None:
        for el in reversed(emitted):
            if el.parent_id != parent_id:
                continue
            if el.legal_numbering and re.search(r"\(\d+\)$", el.legal_numbering):
                return el.legal_numbering
        return None


# ────────────────────────────────────────────────────────────────────────────
# Module-level splitters (testable in isolation).
# ────────────────────────────────────────────────────────────────────────────


def _blocks(text: str) -> list[_LineBlock]:
    """Split source text into blank-line-delimited blocks, preserving offsets."""
    out: list[_LineBlock] = []
    cursor = 0
    for raw in re.split(r"(\n\s*\n)", text):
        if not raw:
            continue
        if raw.strip() == "":
            cursor += len(raw)
            continue
        out.append(_LineBlock(text=raw, char_start=cursor, char_end=cursor + len(raw)))
        cursor += len(raw)
    return out


def _classify_heading(block_text: str) -> str | None:
    """Return ``"chapter" | "section" | "article" | None`` for a block."""
    first = block_text.splitlines()[0].strip()
    if re.match(r"^CHAPTER\s+", first, re.IGNORECASE):
        return "chapter"
    if re.match(r"^(Section|Sec\.?|s\.)\s+\d+", first, re.IGNORECASE):
        return "section"
    if re.match(r"^(Article|Art\.?)\s+\d+", first, re.IGNORECASE):
        return "article"
    return None


def _split_into_paragraphs(body_text: str) -> list[str]:
    """Split a body region into one paragraph per sub-clause / per top-level paragraph.

    A new paragraph starts at a line that begins with ``(N)`` or ``(a)``
    (with optional leading whitespace). Continuation lines (indented
    wrap-arounds) attach to the current paragraph.
    """
    if not body_text:
        return []
    lines = body_text.splitlines(keepends=True)
    out: list[str] = []
    buf: list[str] = []

    def _flush() -> None:
        if buf:
            out.append("".join(buf))
            buf.clear()

    for line in lines:
        stripped = line.lstrip()
        starts_sub = bool(_NUMBERED_SUB.match(stripped) or _LETTERED_SUB.match(stripped))
        if starts_sub and buf:
            _flush()
        buf.append(line)
    _flush()

    # If there are no sub-clause openers at all, the whole body is one paragraph.
    if not out:
        return [body_text]
    return out
