"""ID-replacement citation builder — the anti-hallucination keystone.

The LLM only ever emits a span ID like `doc12#4120-4215`. This module is the
ONLY place where a human-readable citation string is constructed, and it does
so from stored metadata — never from model output.
"""

from __future__ import annotations

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
    assert span.element_id == element.element_id, "span/element mismatch"
    assert element.doc_id == document.doc_id, "element/doc mismatch"
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
