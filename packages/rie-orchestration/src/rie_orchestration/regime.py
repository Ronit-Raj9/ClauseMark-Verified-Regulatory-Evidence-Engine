"""Whole-law regime assembly via ``rie_domain.assemble_regime``."""

from __future__ import annotations

from collections.abc import Sequence

from rie_contracts import Element, LegalRegime, StructureEdge
from rie_domain import assemble_regime

__all__ = ["assemble_regime_for_clause"]


def assemble_regime_for_clause(
    primary: Element,
    elements_by_id: dict[str, Element],
    edges: Sequence[StructureEdge],
    *,
    max_depth: int = 2,
) -> LegalRegime:
    """Walk the structure graph from *primary* to build a ``LegalRegime``."""
    return assemble_regime(primary, elements_by_id, edges, max_depth=max_depth)
