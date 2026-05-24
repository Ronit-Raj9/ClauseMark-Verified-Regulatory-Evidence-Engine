"""Mixed-regulatory profile — statute + registries + notices. Phase 2 stub."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from rie_contracts import DocumentMeta, DocumentProfile, Element, ElementType, StructureEdge


@dataclass
class MixedRegulatoryStrategy:
    profile: DocumentProfile = field(default=DocumentProfile.MIXED_REGULATORY)

    def child_chunk_size(self) -> int:
        return 384

    def child_chunk_overlap(self) -> int:
        return 32

    def applicable_element_types(self) -> frozenset[str]:
        return frozenset(
            {
                ElementType.PARAGRAPH.value,
                ElementType.SECTION.value,
                ElementType.TABLE.value,
                ElementType.LIST_ITEM.value,
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
