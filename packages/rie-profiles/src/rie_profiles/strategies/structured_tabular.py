"""Structured-tabular profile — tariff schedules, HS-code tables. Phase 2 stub."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from rie_contracts import DocumentMeta, DocumentProfile, Element, ElementType, StructureEdge


@dataclass
class StructuredTabularStrategy:
    profile: DocumentProfile = field(default=DocumentProfile.STRUCTURED_TABULAR)

    def child_chunk_size(self) -> int:
        return 256

    def child_chunk_overlap(self) -> int:
        return 0

    def applicable_element_types(self) -> frozenset[str]:
        return frozenset({ElementType.TABLE.value, ElementType.TABLE_CELL.value})

    def supports_llm_classification(self) -> bool:
        return False

    def post_extract(
        self,
        meta: DocumentMeta,
        elements: Sequence[Element],
        edges: Sequence[StructureEdge],
    ) -> tuple[Sequence[Element], Sequence[StructureEdge]]:
        return elements, edges
