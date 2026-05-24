"""Statutory legal text profile — Acts, statutes, amendments. Built."""

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


@dataclass
class StatutoryLegalTextStrategy:
    """Hexagonal-style legal-text strategy.

    Child chunks are small (~480 tokens equivalent) so similarity matches a
    single rule or definition; parent context expands to the whole element +
    structure-graph neighbourhood at retrieval time.
    """

    profile: DocumentProfile = field(default=DocumentProfile.STATUTORY_LEGAL_TEXT)

    def child_chunk_size(self) -> int:
        return 480

    def child_chunk_overlap(self) -> int:
        return 64

    def applicable_element_types(self) -> frozenset[str]:
        return frozenset(
            {
                ElementType.ARTICLE.value,
                ElementType.SECTION.value,
                ElementType.PARAGRAPH.value,
                ElementType.LIST_ITEM.value,
                ElementType.DEFINITION.value,
                ElementType.SCHEDULE.value,
            }
        )

    def supports_llm_classification(self) -> bool:
        return True

    def post_extract(
        self,
        meta: DocumentMeta,
        elements: Sequence[Element],
        edges: Sequence[StructureEdge],
    ) -> tuple[Sequence[Element], Sequence[StructureEdge]]:
        return elements, edges
