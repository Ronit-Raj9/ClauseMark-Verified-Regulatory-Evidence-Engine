"""Deterministic citation materialisation via ``rie_domain.build_citation``.

The API and other callers import from here so ``build_citation`` stays the
single construction point while respecting package dependency direction.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from rie_contracts import DocumentMeta, Element, EvidenceSpan
from rie_domain import build_citation
from rie_domain.citation import Citation

__all__ = ["build_citation", "extract_snippet", "materialise_citations"]


def extract_snippet(element_text: str, span: EvidenceSpan) -> str:
    """Slice stored element text by verified char offsets; fall back to full text."""
    if not element_text:
        return ""
    snippet = element_text[span.char_start : span.char_end]
    return snippet or element_text


def materialise_citations(
    spans: Sequence[EvidenceSpan],
    *,
    get_element_text: Callable[[str], str],
    get_element: Callable[[str], Element],
    get_document: Callable[[str], DocumentMeta],
) -> list[Citation]:
    """Resolve span IDs to domain ``Citation`` objects — never LLM-authored."""
    out: list[Citation] = []
    for span in spans:
        try:
            element = get_element(span.element_id)
            document = get_document(span.doc_id)
            text = get_element_text(span.element_id)
        except KeyError:
            continue
        snippet = extract_snippet(text, span)
        out.append(build_citation(span, element, document, snippet))
    return out
