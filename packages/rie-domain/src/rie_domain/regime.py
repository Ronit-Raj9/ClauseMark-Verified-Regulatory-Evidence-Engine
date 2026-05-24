"""Whole-law regime assembly + self-consistency tally."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence

from rie_contracts import (
    Element,
    Layer1Status,
    LegalRegime,
    StructureEdge,
    StructureEdgeType,
)

REGIME_EDGE_TYPES: frozenset[StructureEdgeType] = frozenset(
    {
        StructureEdgeType.CROSS_REFERENCE,
        StructureEdgeType.DEFINES,
        StructureEdgeType.PROVISO,
        StructureEdgeType.NOTWITHSTANDING,
    }
)


def assemble_regime(
    primary: Element,
    elements_by_id: dict[str, Element],
    edges: Sequence[StructureEdge],
    max_depth: int = 2,
) -> LegalRegime:
    """Walk the structure graph from a primary clause to assemble its regime.

    `max_depth` bounds graph traversal so a transitive web of references does
    not explode context. Definitions and provisos are tracked separately so
    the classifier and verifier can weight them deliberately.
    """

    seen: set[str] = {primary.element_id}
    defs: list[str] = []
    exceptions: list[str] = []
    members: list[str] = [primary.element_id]

    frontier = [primary.element_id]
    for _ in range(max_depth):
        next_frontier: list[str] = []
        for src_id in frontier:
            for edge in edges:
                if edge.from_element != src_id and edge.to_element != src_id:
                    continue
                if edge.edge_type not in REGIME_EDGE_TYPES:
                    continue
                other = edge.to_element if edge.from_element == src_id else edge.from_element
                if other in seen or other not in elements_by_id:
                    continue
                seen.add(other)
                members.append(other)
                next_frontier.append(other)
                if edge.edge_type == StructureEdgeType.DEFINES:
                    defs.append(other)
                elif edge.edge_type in {
                    StructureEdgeType.PROVISO,
                    StructureEdgeType.NOTWITHSTANDING,
                }:
                    exceptions.append(other)
        frontier = next_frontier
        if not frontier:
            break

    return LegalRegime(
        primary_element_id=primary.element_id,
        member_element_ids=members,
        definitions=defs,
        exceptions=exceptions,
    )


def consolidate_self_consistency_votes(
    indicator_ids: Iterable[str],
) -> tuple[str, dict[str, int]]:
    """Majority-vote with tie-break by lexicographic order."""

    counts = Counter(indicator_ids)
    if not counts:
        return ("", {})
    top = max(counts.items(), key=lambda kv: (kv[1], -ord(kv[0][0]) if kv[0] else 0))
    return (top[0], dict(counts))


def derive_layer1_status(
    classification_passed: bool, self_consistency_unstable: bool
) -> Layer1Status:
    if not classification_passed:
        return Layer1Status.REJECTED
    if self_consistency_unstable:
        return Layer1Status.FLAGGED
    return Layer1Status.PENDING_VERIFICATION
