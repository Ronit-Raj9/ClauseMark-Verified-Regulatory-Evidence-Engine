"""Build cross-reference / definition / proviso / notwithstanding edges.

Implements `rie_contracts.StructureGraphBuilderPort`. Walks the ordered
element stream produced by an extractor and emits `StructureEdge` rows.
Resolution is deterministic — regex-driven over legal numbering and
fixed clause-opening tokens. The LLM never touches this stage.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from rie_contracts import (
    Element,
    ElementType,
    StructureEdge,
    StructureEdgeType,
)

from rie_extract.structure.legal_numbering import find_cross_references

# Headings that flag a definitions block.
_DEF_HEADING = re.compile(r"\b(definitions?|interpretation)\b", re.IGNORECASE)

# A definition typically opens with a quoted term: `"personal data" means …`.
_DEF_TERM = re.compile(r'["“]([^"”]{2,80})["”]\s+means\b', re.IGNORECASE)

# Proviso opening tokens. Matches at the start of a paragraph OR after a
# heading prefix like "Section 7." so we catch both standalone provisos
# and provisos inline in a numbered clause.
_PROVISO = re.compile(
    r"(?:^|\.\s+)(?:provided\s+that\b|provided\s+further\s+that\b|provided\s+also\s+that\b)",
    re.IGNORECASE,
)

# Exception-style proviso: "(N) Subsection (M) does not apply where…"
_PROVISO_EXCEPTION = re.compile(
    r"\bsub-?section\s*\(\d+\)\s+does\s+not\s+apply\b",
    re.IGNORECASE,
)

# Notwithstanding opener — accept after a heading prefix too.
_NOTWITHSTANDING = re.compile(r"(?:^|\.\s+)notwithstanding\b", re.IGNORECASE)


@dataclass
class StructureGraphBuilder:
    """Deterministic structure-graph builder.

    Edge kinds emitted:
      • DEFINES — every element under a "Definitions"/"Interpretation"
        section whose text matches the ``"term" means …`` pattern is
        linked, by emitting one edge per *referencing* element whose
        text contains the defined term.
      • CROSS_REFERENCE — for each cross-reference in element text,
        resolve target by `legal_numbering` exact match.
      • PROVISO — clauses opening with "Provided that"; also exception
        sub-paragraphs of the "Subsection (N) does not apply" form. The
        proviso is linked back to its parent / preceding numbered
        sibling.
      • NOTWITHSTANDING — clauses opening with "Notwithstanding"; linked
        to the cross-referenced target where one is present, otherwise
        to the parent element.
    """

    def build(self, elements: Sequence[Element]) -> Sequence[StructureEdge]:
        edges: list[StructureEdge] = []

        by_numbering = self._index_by_numbering(elements)
        by_id = {el.element_id: el for el in elements}

        edges.extend(self._cross_reference_edges(elements, by_numbering))
        edges.extend(self._definition_edges(elements))
        edges.extend(self._proviso_edges(elements, by_id))
        edges.extend(self._notwithstanding_edges(elements, by_numbering, by_id))

        # De-dupe while preserving order — edges are frozen models so
        # they hash naturally.
        seen: set[tuple[str, str, str]] = set()
        unique: list[StructureEdge] = []
        for e in edges:
            key = (e.from_element, e.to_element, e.edge_type.value)
            if key in seen:
                continue
            seen.add(key)
            unique.append(e)
        return unique

    # ── Edge builders ───────────────────────────────────────────────────

    def _cross_reference_edges(
        self,
        elements: Sequence[Element],
        by_numbering: dict[str, str],
    ) -> list[StructureEdge]:
        out: list[StructureEdge] = []
        for el in elements:
            if not el.text:
                continue
            for raw, canonical in find_cross_references(el.text):
                target = self._resolve_reference(el, canonical, by_numbering)
                if target is None or target == el.element_id:
                    continue
                out.append(
                    StructureEdge(
                        from_element=el.element_id,
                        to_element=target,
                        edge_type=StructureEdgeType.CROSS_REFERENCE,
                        raw_reference=raw,
                    )
                )
        return out

    def _definition_edges(self, elements: Sequence[Element]) -> list[StructureEdge]:
        defined_terms: dict[str, str] = {}  # term lower → defining element id

        in_definitions_block = False
        current_section_is_defs = False
        for el in elements:
            if el.element_type in {ElementType.HEADING, ElementType.SECTION, ElementType.ARTICLE}:
                current_section_is_defs = bool(_DEF_HEADING.search(el.text))
                in_definitions_block = current_section_is_defs
            if not in_definitions_block:
                continue
            # Treat any child element whose text introduces a quoted term as a definition.
            m = _DEF_TERM.search(el.text)
            if m:
                term = m.group(1).strip().lower()
                if term and term not in defined_terms:
                    defined_terms[term] = el.element_id

        if not defined_terms:
            return []

        # For each non-definition element, link every defined term it mentions.
        out: list[StructureEdge] = []
        for el in elements:
            if el.element_id in defined_terms.values():
                continue
            lowered = el.text.lower()
            for term, defining_id in defined_terms.items():
                if term in lowered:
                    out.append(
                        StructureEdge(
                            from_element=defining_id,
                            to_element=el.element_id,
                            edge_type=StructureEdgeType.DEFINES,
                            raw_reference=term,
                        )
                    )
        return out

    def _proviso_edges(
        self,
        elements: Sequence[Element],
        by_id: dict[str, Element],
    ) -> list[StructureEdge]:
        out: list[StructureEdge] = []
        for el in elements:
            is_proviso = bool(_PROVISO.search(el.text)) or bool(_PROVISO_EXCEPTION.search(el.text))
            if not is_proviso:
                continue
            target_id = el.parent_id or self._previous_sibling_id(el, elements)
            if target_id and target_id in by_id and target_id != el.element_id:
                out.append(
                    StructureEdge(
                        from_element=el.element_id,
                        to_element=target_id,
                        edge_type=StructureEdgeType.PROVISO,
                        raw_reference=el.text[:80].strip(),
                    )
                )
        return out

    def _notwithstanding_edges(
        self,
        elements: Sequence[Element],
        by_numbering: dict[str, str],
        by_id: dict[str, Element],
    ) -> list[StructureEdge]:
        out: list[StructureEdge] = []
        for el in elements:
            if not _NOTWITHSTANDING.search(el.text):
                continue
            target: str | None = None
            for _raw, canonical in find_cross_references(el.text):
                resolved = self._resolve_reference(el, canonical, by_numbering)
                if resolved and resolved != el.element_id:
                    target = resolved
                    break
            if target is None and el.parent_id and el.parent_id in by_id:
                target = el.parent_id
            if target is None:
                continue
            out.append(
                StructureEdge(
                    from_element=el.element_id,
                    to_element=target,
                    edge_type=StructureEdgeType.NOTWITHSTANDING,
                    raw_reference=el.text[:80].strip(),
                )
            )
        return out

    # ── Helpers ─────────────────────────────────────────────────────────

    def _index_by_numbering(self, elements: Sequence[Element]) -> dict[str, str]:
        """Map canonical numbering → element_id. First writer wins."""
        out: dict[str, str] = {}
        for el in elements:
            if el.legal_numbering and el.legal_numbering not in out:
                out[el.legal_numbering] = el.element_id
        return out

    def _resolve_reference(
        self,
        source: Element,
        canonical: str,
        by_numbering: dict[str, str],
    ) -> str | None:
        """Resolve a canonical reference to an element id, with relative fall-back.

        Handles bare "(subsection N)" references by composing them with
        the source element's own section number, e.g. inside ``s. 26(2)``
        a mention of ``subsection (1)`` resolves to ``s. 26(1)``.
        """
        if canonical in by_numbering:
            return by_numbering[canonical]

        # Relative sub-section reference of the form "(subsection (N))" or
        # "(subsection (N)(a))".
        sub_match = re.match(r"\(subsection \((\d+)\)(?:\(([a-z]+)\))?\)", canonical)
        if sub_match and source.legal_numbering:
            sub_num = sub_match.group(1)
            para = sub_match.group(2)
            section_match = re.match(r"(s\.|art\.)\s+(\d+[A-Za-z]?)", source.legal_numbering)
            if section_match:
                synthetic = f"{section_match.group(1)} {section_match.group(2)}({sub_num})"
                if para:
                    synthetic += f"({para})"
                if synthetic in by_numbering:
                    return by_numbering[synthetic]
        return None

    def _previous_sibling_id(self, target: Element, elements: Sequence[Element]) -> str | None:
        prev: str | None = None
        for el in elements:
            if el.element_id == target.element_id:
                return prev
            if el.parent_id == target.parent_id:
                prev = el.element_id
        return prev
