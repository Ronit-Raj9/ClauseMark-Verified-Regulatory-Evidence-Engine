"""Base protocol every document-profile strategy implements."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from rie_contracts import DocumentMeta, DocumentProfile, Element, StructureEdge


class ProfileError(RuntimeError):
    pass


@runtime_checkable
class DocumentProfileStrategy(Protocol):
    """Strategy contract — ingest/extract/chunk behaviour for a profile family.

    A profile says how evidence is SHAPED. Implementations decide:
      • how to split a document into addressable elements
      • what counts as a structure edge in this profile
      • what child-chunk size makes sense for retrieval (small enough to be
        precise, big enough to mean something)
      • whether evaluation should hit the LLM at all (treaty lookups skip it)
    """

    profile: DocumentProfile

    def child_chunk_size(self) -> int: ...

    def child_chunk_overlap(self) -> int: ...

    def applicable_element_types(self) -> frozenset[str]: ...

    def supports_llm_classification(self) -> bool: ...

    def post_extract(
        self,
        meta: DocumentMeta,
        elements: Sequence[Element],
        edges: Sequence[StructureEdge],
    ) -> tuple[Sequence[Element], Sequence[StructureEdge]]:
        """Hook for profile-specific post-processing. Default: identity."""
        ...
