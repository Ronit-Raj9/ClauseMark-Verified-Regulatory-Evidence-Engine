"""ID-replacement citation builder — the anti-hallucination keystone.

The LLM only ever emits a span ID like `doc12#4120-4215`. This module is the
ONLY place where a human-readable citation string is constructed, and it does
so from stored metadata — never from model output.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from rie_contracts import DocumentMeta, Element, EvidenceSpan


@dataclass(frozen=True)
class Citation:
    span_id: str
    doc_id: str
    document_title: str
    legal_locator: str  # e.g. "Art. 26(1)" or "s. 8(1)"
    page: int
    snippet: str
    source_url: str | None
    effective_date: str | None
    char_start: int
    char_end: int


def build_citation(
    span: EvidenceSpan,
    element: Element,
    document: DocumentMeta,
    snippet: str,
) -> Citation:
    """Materialise a citation from stored metadata (§6.6 ID-replacement).

    The LLM emits only ``span_id``; this function is the sole construction
    point for human-readable citation fields. Raises ``ValueError`` when span,
    element, or document identifiers are inconsistent.
    """
    if span.element_id != element.element_id:
        msg = f"span/element mismatch: {span.element_id!r} != {element.element_id!r}"
        raise ValueError(msg)
    if element.doc_id != document.doc_id:
        msg = f"element/doc mismatch: {element.doc_id!r} != {document.doc_id!r}"
        raise ValueError(msg)
    locator = element.legal_numbering or f"page {element.page}"
    return Citation(
        span_id=span.span_id,
        doc_id=document.doc_id,
        document_title=document.title,
        legal_locator=locator,
        page=element.page,
        snippet=snippet,
        source_url=document.source_url,
        effective_date=document.effective_date.isoformat() if document.effective_date else None,
        char_start=span.char_start,
        char_end=span.char_end,
    )


def render_citation_string(c: Citation) -> str:
    base = f"{c.document_title}, {c.legal_locator} (p. {c.page})"
    if c.effective_date:
        base += f" [eff. {c.effective_date}]"
    return base


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
    """Resolve span IDs to ``Citation`` objects — never LLM-authored."""
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
