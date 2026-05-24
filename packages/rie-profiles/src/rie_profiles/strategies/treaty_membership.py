"""Treaty-membership profile — membership lookup, not text extraction. Phase 2 stub."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from rie_contracts import DocumentMeta, DocumentProfile, Element, StructureEdge


@dataclass
class TreatyMembershipStrategy:
    profile: DocumentProfile = field(default=DocumentProfile.TREATY_MEMBERSHIP)

    def child_chunk_size(self) -> int:
        return 0  # no chunking — treaty lookup is membership not text

    def child_chunk_overlap(self) -> int:
        return 0

    def applicable_element_types(self) -> frozenset[str]:
        return frozenset()

    def supports_llm_classification(self) -> bool:
        return False

    def post_extract(
        self,
        meta: DocumentMeta,
        elements: Sequence[Element],
        edges: Sequence[StructureEdge],
    ) -> tuple[Sequence[Element], Sequence[StructureEdge]]:
        return elements, edges
