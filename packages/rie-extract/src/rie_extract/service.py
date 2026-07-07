"""Top-level extraction service — implements `DocumentExtractorPort`.

Routes to the right concrete extractor, runs structure-graph construction,
and applies the document profile's ``post_extract`` hook so profile-
specific normalisation can run without spilling into engine code.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from rie_contracts import (
    DocumentMeta,
    DocumentProfile,
    Element,
    StructureEdge,
)
from rie_profiles import DocumentProfileStrategy, get_strategy

from rie_extract.router import Extractor, pick_adapter
from rie_extract.structure.graph_builder import StructureGraphBuilder


@dataclass
class ExtractionService:
    """Concrete `DocumentExtractorPort` implementation."""

    graph_builder: StructureGraphBuilder = field(default_factory=StructureGraphBuilder)
    default_profile: DocumentProfile = DocumentProfile.STATUTORY_LEGAL_TEXT

    def extract(
        self,
        doc_meta: DocumentMeta,
        raw: bytes,
    ) -> tuple[Sequence[Element], Sequence[StructureEdge]]:
        adapter: Extractor = pick_adapter(doc_meta, raw=raw)
        elements, adapter_edges = adapter.extract(doc_meta, raw)

        # Graph builder runs over whatever the adapter produced; edges from
        # the adapter (rare) are preserved.
        graph_edges = self.graph_builder.build(elements)
        edges: list[StructureEdge] = list(adapter_edges) + list(graph_edges)

        # Profile hook — defaults to identity for statutory_legal_text.
        strategy: DocumentProfileStrategy = self._resolve_strategy()
        elements, edges = strategy.post_extract(doc_meta, elements, edges)
        return elements, edges

    def extract_path(
        self,
        doc_meta: DocumentMeta,
        path: str | Path,
    ) -> tuple[Sequence[Element], Sequence[StructureEdge]]:
        """Convenience for callers that already have a file on disk."""
        p = Path(path)
        raw = p.read_bytes()
        adapter: Extractor = pick_adapter(doc_meta, source_path=p)
        elements, adapter_edges = adapter.extract(doc_meta, raw)
        graph_edges = self.graph_builder.build(elements)
        edges: list[StructureEdge] = list(adapter_edges) + list(graph_edges)
        strategy: DocumentProfileStrategy = self._resolve_strategy()
        elements, edges = strategy.post_extract(doc_meta, elements, edges)
        return elements, edges

    def _resolve_strategy(self) -> DocumentProfileStrategy:
        return get_strategy(self.default_profile)
