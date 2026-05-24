"""Treaty-membership profile — membership lookup, not text extraction.

post_extract collapses the document into a single "membership-row" Element
keyed by jurisdiction. The orchestrator's tabular_lookup branch treats this
single row as the addressable unit for treaty membership questions
(party / non-party / signatory). There is no clause-extraction LLM call on
this profile; the document is essentially a row in a treaty-membership table.

Returns a minimal element list (one element) for the orchestrator's
tabular_lookup branch. Original `edges` are dropped because the membership
row has no internal structure to reference.
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


@dataclass
class TreatyMembershipStrategy:
    """Membership-lookup strategy. No chunking, no LLM."""

    profile: DocumentProfile = field(default=DocumentProfile.TREATY_MEMBERSHIP)

    def child_chunk_size(self) -> int:
        return 0  # no chunking — treaty lookup is membership not text

    def child_chunk_overlap(self) -> int:
        return 0

    def applicable_element_types(self) -> frozenset[str]:
        # Single OTHER element acts as the membership row.
        return frozenset({ElementType.OTHER.value})

    def supports_llm_classification(self) -> bool:
        return False

    def post_extract(
        self,
        meta: DocumentMeta,
        elements: Sequence[Element],
        edges: Sequence[StructureEdge],
    ) -> tuple[Sequence[Element], Sequence[StructureEdge]]:
        # Collapse to a single jurisdiction-keyed membership row.
        # Concatenate any extracted text so it remains addressable (e.g. for
        # downstream "evidence_found" coverage); offsets span the joined text.
        joined = "\n".join(e.text for e in elements if e.text)
        if not joined:
            joined = meta.title  # fall back to doc title so the row has content
        row_id = f"{meta.doc_id}#membership:{meta.jurisdiction}"
        row = Element(
            element_id=row_id,
            doc_id=meta.doc_id,
            parent_id=None,
            element_type=ElementType.OTHER,
            text=joined,
            page=1,
            char_start=0,
            char_end=len(joined),
            extraction_confidence=1.0,
            legal_numbering=f"membership:{meta.jurisdiction}",
        )
        # Drop original structure edges — collapsed row has no internal refs.
        return [row], []
