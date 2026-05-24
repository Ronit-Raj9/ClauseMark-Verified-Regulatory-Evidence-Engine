"""Mixed-regulatory profile — statute + registries + notices.

Post-extract tags each element with a routing hint so the orchestrator can
fan-out: paragraphs / sections / list-items go to the LLM clause-extraction
path, tables go to the tabular-lookup path.

The contract's `Element` model is frozen (no free-form metadata field), so the
routing hint is written into `legal_numbering` as a `route:<branch>` prefix
when no legal numbering exists, or appended when it does. Downstream code can
parse the prefix; absence of a hint means "default to clause_extraction".
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

_ROUTE_PREFIX = "route:"
_ROUTE_LLM = "clause_extraction"
_ROUTE_TABULAR = "tabular_lookup"

_LLM_TYPES: frozenset[ElementType] = frozenset(
    {
        ElementType.PARAGRAPH,
        ElementType.SECTION,
        ElementType.ARTICLE,
        ElementType.LIST_ITEM,
        ElementType.DEFINITION,
        ElementType.SCHEDULE,
    }
)
_TABULAR_TYPES: frozenset[ElementType] = frozenset({ElementType.TABLE, ElementType.TABLE_CELL})


def _route_for(et: ElementType) -> str | None:
    if et in _LLM_TYPES:
        return _ROUTE_LLM
    if et in _TABULAR_TYPES:
        return _ROUTE_TABULAR
    return None


def _tag(existing: str | None, route: str) -> str:
    hint = f"{_ROUTE_PREFIX}{route}"
    if existing is None or existing == "":
        return hint
    if _ROUTE_PREFIX in existing:
        # Don't double-tag; leave caller-supplied hint in place.
        return existing
    return f"{existing}|{hint}"


@dataclass
class MixedRegulatoryStrategy:
    """Statute-plus-tables strategy. LLM clause-extraction for prose, lookup for tables."""

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
                ElementType.ARTICLE.value,
                ElementType.LIST_ITEM.value,
                ElementType.DEFINITION.value,
                ElementType.TABLE.value,
                ElementType.TABLE_CELL.value,
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
        tagged: list[Element] = []
        for el in elements:
            route = _route_for(el.element_type)
            if route is None:
                tagged.append(el)
                continue
            updated = el.model_copy(update={"legal_numbering": _tag(el.legal_numbering, route)})
            tagged.append(updated)
        return tagged, edges
