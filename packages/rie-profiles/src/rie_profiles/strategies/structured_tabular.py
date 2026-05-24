"""Structured-tabular profile — tariff schedules, HS-code tables.

Post-extract:
  - Detect TABLE elements.
  - If a table has no child rows/cells already attached (via `parent_id`),
    split its `text` into rows (newline-separated) and cells (tab- or
    pipe-separated) and emit those as new Elements parented to the table.
  - Hierarchy is materialised through `Element.parent_id`. `StructureEdge`
    is reserved for semantic relations (cross_reference / defines / proviso /
    amends / notwithstanding), so we DO NOT fabricate a CONTAINS edge here;
    see report-note in package docs.

Note (contract-gap, escalation candidate):
  - `ElementType.TABLE_ROW` does not exist in the frozen contracts. We use
    `ElementType.OTHER` with `legal_numbering="row:<N>"` for rows so the row
    layer is still addressable. Cells use `ElementType.TABLE_CELL`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from rie_contracts import (
    DocumentMeta,
    DocumentProfile,
    Element,
    ElementType,
    StructureEdge,
)

# Cell separators accepted in raw table text (most-specific first).
_CELL_SEPARATORS: tuple[str, ...] = ("\t", " | ", "|")


def _split_row(line: str) -> list[str]:
    for sep in _CELL_SEPARATORS:
        if sep in line:
            return [c.strip() for c in line.split(sep) if c.strip()]
    # Single-column row.
    stripped = line.strip()
    return [stripped] if stripped else []


@dataclass
class StructuredTabularStrategy:
    """Tariff-schedule / HS-code strategy. Lookup-only, not LLM clause-extraction."""

    profile: DocumentProfile = field(default=DocumentProfile.STRUCTURED_TABULAR)

    def child_chunk_size(self) -> int:
        # Small chunks — one row should match precisely, not bleed into neighbours.
        return 256

    def child_chunk_overlap(self) -> int:
        return 0

    def applicable_element_types(self) -> frozenset[str]:
        return frozenset(
            {
                ElementType.TABLE.value,
                ElementType.TABLE_CELL.value,
                # Rows currently fall back to OTHER (no TABLE_ROW in contract).
                ElementType.OTHER.value,
            }
        )

    def supports_llm_classification(self) -> bool:
        # Tabular lookup path. No clause-extraction LLM call.
        return False

    def post_extract(
        self,
        meta: DocumentMeta,
        elements: Sequence[Element],
        edges: Sequence[StructureEdge],
    ) -> tuple[Sequence[Element], Sequence[StructureEdge]]:
        existing_children: set[str] = {e.parent_id for e in elements if e.parent_id is not None}
        new_elements: list[Element] = list(elements)

        for table in elements:
            if table.element_type is not ElementType.TABLE:
                continue
            if table.element_id in existing_children:
                # Table already has rows/cells attached; respect upstream split.
                continue

            cursor = table.char_start
            row_idx = 0
            for line in table.text.splitlines():
                cells = _split_row(line)
                if not cells:
                    cursor += len(line) + 1  # +1 for the newline
                    continue
                row_start = cursor
                row_end = cursor + len(line)
                row_id = f"{table.element_id}.r{row_idx}"
                row_el = Element(
                    element_id=row_id,
                    doc_id=table.doc_id,
                    parent_id=table.element_id,
                    element_type=ElementType.OTHER,
                    text=line,
                    page=table.page,
                    bbox=table.bbox,
                    char_start=row_start,
                    char_end=row_end,
                    extraction_confidence=table.extraction_confidence,
                    ocr_engine=table.ocr_engine,
                    legal_numbering=f"row:{row_idx}",
                )
                new_elements.append(row_el)

                # Cell offsets are best-effort: locate each cell text inside the line.
                local = 0
                for cell_idx, cell_text in enumerate(cells):
                    found = line.find(cell_text, local)
                    if found < 0:
                        found = local
                    c_start = row_start + found
                    c_end = c_start + len(cell_text)
                    local = found + len(cell_text)
                    cell_id = f"{row_id}.c{cell_idx}"
                    new_elements.append(
                        Element(
                            element_id=cell_id,
                            doc_id=table.doc_id,
                            parent_id=row_id,
                            element_type=ElementType.TABLE_CELL,
                            text=cell_text,
                            page=table.page,
                            bbox=table.bbox,
                            char_start=c_start,
                            char_end=c_end,
                            extraction_confidence=table.extraction_confidence,
                            ocr_engine=table.ocr_engine,
                            legal_numbering=f"row:{row_idx};col:{cell_idx}",
                        )
                    )
                row_idx += 1
                cursor = row_end + 1  # advance past newline

        return new_elements, edges
