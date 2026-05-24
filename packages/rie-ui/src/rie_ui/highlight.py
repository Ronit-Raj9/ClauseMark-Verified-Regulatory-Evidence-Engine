"""Span-highlight rendering for the claim-detail page.

Pure function: ``render_with_highlight(doc_text, spans)`` returns an HTML
string with ``<mark>`` tags around each cited span. Overlapping or adjacent
spans are merged so a character is never double-wrapped. All non-span text
is HTML-escaped — the function is safe to feed into ``st.markdown(...,
unsafe_allow_html=True)`` or ``st.html``.
"""

from __future__ import annotations

from collections.abc import Iterable
from html import escape

# Public type alias for clarity at call sites.
Span = tuple[int, int]


def _normalise_spans(spans: Iterable[Span], doc_len: int) -> list[Span]:
    """Clamp to the document, drop empties, sort, then merge overlaps.

    Spans that touch (``a.end == b.start``) are also merged — a visible
    seam between two ``<mark>`` blocks adds nothing.
    """
    cleaned: list[Span] = []
    for raw in spans:
        if not isinstance(raw, tuple) or len(raw) != 2:
            raise TypeError(f"span must be a (start, end) tuple, got {raw!r}")
        start, end = int(raw[0]), int(raw[1])
        if end < start:
            raise ValueError(f"span end {end} < start {start}")
        start = max(0, min(start, doc_len))
        end = max(0, min(end, doc_len))
        if end <= start:
            continue
        cleaned.append((start, end))

    if not cleaned:
        return []

    cleaned.sort()
    merged: list[Span] = [cleaned[0]]
    for start, end in cleaned[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def render_with_highlight(doc_text: str, spans: Iterable[Span]) -> str:
    """Return HTML with ``<mark>`` around each (possibly merged) span.

    - Non-span text is HTML-escaped.
    - Overlapping / adjacent spans are merged so we never emit nested marks.
    - Newlines are preserved (callers can wrap output in a ``<pre>`` block
      or use ``white-space: pre-wrap`` to render them).
    """
    if not isinstance(doc_text, str):
        raise TypeError(f"doc_text must be str, got {type(doc_text).__name__}")

    normalised = _normalise_spans(spans, len(doc_text))
    if not normalised:
        return escape(doc_text)

    parts: list[str] = []
    cursor = 0
    for start, end in normalised:
        if start > cursor:
            parts.append(escape(doc_text[cursor:start]))
        parts.append("<mark>")
        parts.append(escape(doc_text[start:end]))
        parts.append("</mark>")
        cursor = end
    if cursor < len(doc_text):
        parts.append(escape(doc_text[cursor:]))
    return "".join(parts)


__all__ = ["Span", "render_with_highlight"]
