"""Deterministic grammar over legal numbering.

Cross-reference resolution is deterministic — regex + grammar over the
small but stable vocabulary statutes use ("Section 26", "s. 26(1)",
"Article 12", "subsection (3)", "paragraph (a)"). We never ask the LLM
to find references; that would re-introduce hallucination at a layer
that has no business being uncertain.

Canonical form is `s. <num>[<letter>][(<sub>)][(<para>)]`, e.g.
``s. 26(1)(a)``. Article forms canonicalise to ``art. <num>...``.
"""

from __future__ import annotations

import re
from re import Pattern

# ────────────────────────────────────────────────────────────────────────────
# Primary heading patterns — anchored to start of a line / heading text.
# ────────────────────────────────────────────────────────────────────────────

# "Section 26", "Section 26A", "s. 12", "Sec 5", "SECTION 30"
_SECTION: Pattern[str] = re.compile(
    r"\b(?:Section|Sec\.?|s\.)\s+(\d+[A-Za-z]?)",
    re.IGNORECASE,
)

# "Article 12", "Article 12A", "Art. 12", "ART 12"
_ARTICLE: Pattern[str] = re.compile(
    r"\b(?:Article|Art\.?)\s+(\d+[A-Za-z]?)",
    re.IGNORECASE,
)

# "CHAPTER I", "Chapter 4", "CHAPTER VI"
_CHAPTER: Pattern[str] = re.compile(
    r"\bCHAPTER\s+([IVXLCDM]+|\d+)",
    re.IGNORECASE,
)

# Sub-section number "(1)" / "(12)"
_SUBSECTION: Pattern[str] = re.compile(r"\((\d+)\)")

# Paragraph letter "(a)" / "(ii)"
_PARAGRAPH: Pattern[str] = re.compile(r"\(([a-z]+)\)")

# ────────────────────────────────────────────────────────────────────────────
# Cross-reference patterns (used inside running text).
# ────────────────────────────────────────────────────────────────────────────

# "section 12", "section 26(1)", "section 26(1)(a)"
_XREF_SECTION: Pattern[str] = re.compile(
    r"\b(?:section|sec\.?|s\.)\s+(\d+[A-Za-z]?)((?:\([\w]+\))*)",
    re.IGNORECASE,
)

# "subsection (1)" / "sub-section (2)(a)"
_XREF_SUBSECTION: Pattern[str] = re.compile(
    r"\bsub-?section\s+(\(\d+\)(?:\([\w]+\))*)",
    re.IGNORECASE,
)

# "Article 26", "Art. 12(1)"
_XREF_ARTICLE: Pattern[str] = re.compile(
    r"\b(?:article|art\.?)\s+(\d+[A-Za-z]?)((?:\([\w]+\))*)",
    re.IGNORECASE,
)

# "paragraph (a)" / "paragraphs (a) and (b)"
_XREF_PARAGRAPH: Pattern[str] = re.compile(
    r"\bparagraph\s+(\([a-z]+\))",
    re.IGNORECASE,
)


def canonicalize(
    prefix: str, number: str, *, sub: str | None = None, para: str | None = None
) -> str:
    """Build a canonical "s. 26(1)(a)" / "art. 12" identifier."""
    base = f"{prefix.strip().lower()} {number}"
    if sub:
        base += f"({sub})"
    if para:
        base += f"({para})"
    return base


def parse_numbering(heading_text: str) -> str | None:
    """Return canonical numbering for a heading line, or None if not a numbered heading.

    Accepts forms used in statutes/regulations: ``Section 26.``,
    ``Section 26A — Transfer …``, ``Article 12(1)``, ``Art. 5``,
    ``CHAPTER IV — CROSS-BORDER …``.
    """
    if not heading_text:
        return None
    text = heading_text.strip()

    m = _SECTION.search(text)
    if m:
        number = m.group(1)
        # Try to pick up sub-section / paragraph if attached.
        tail = text[m.end() :]
        sub_m = _SUBSECTION.match(tail)
        sub = sub_m.group(1) if sub_m else None
        para = None
        if sub_m:
            after_sub = tail[sub_m.end() :]
            para_m = _PARAGRAPH.match(after_sub)
            para = para_m.group(1) if para_m else None
        return canonicalize("s.", number, sub=sub, para=para)

    m = _ARTICLE.search(text)
    if m:
        number = m.group(1)
        tail = text[m.end() :]
        sub_m = _SUBSECTION.match(tail)
        sub = sub_m.group(1) if sub_m else None
        para = None
        if sub_m:
            after_sub = tail[sub_m.end() :]
            para_m = _PARAGRAPH.match(after_sub)
            para = para_m.group(1) if para_m else None
        return canonicalize("art.", number, sub=sub, para=para)

    m = _CHAPTER.search(text)
    if m:
        return f"ch. {m.group(1).upper()}"

    return None


def find_cross_references(text: str) -> list[tuple[str, str]]:
    """Return a list of (raw_reference, canonical_reference) tuples found in text.

    The canonical form is suitable for matching against an element's
    `legal_numbering` field. Deduplicates by canonical form while
    preserving first-occurrence order.
    """
    if not text:
        return []

    seen: set[str] = set()
    out: list[tuple[str, str]] = []

    def _emit(raw: str, canonical: str) -> None:
        if canonical in seen:
            return
        seen.add(canonical)
        out.append((raw, canonical))

    for m in _XREF_SECTION.finditer(text):
        number = m.group(1)
        tail = m.group(2) or ""
        sub, para = _split_sub_para(tail)
        _emit(m.group(0), canonicalize("s.", number, sub=sub, para=para))

    for m in _XREF_ARTICLE.finditer(text):
        number = m.group(1)
        tail = m.group(2) or ""
        sub, para = _split_sub_para(tail)
        _emit(m.group(0), canonicalize("art.", number, sub=sub, para=para))

    for m in _XREF_SUBSECTION.finditer(text):
        tail = m.group(1)
        sub, para = _split_sub_para(tail)
        # Bare "subsection (X)" → unresolved by itself; mark canonical as a
        # sub-reference. Callers (graph builder) resolve to enclosing section.
        if sub is None:
            continue
        canonical = f"(subsection ({sub}))"
        if para:
            canonical = f"(subsection ({sub})({para}))"
        _emit(m.group(0), canonical)

    for m in _XREF_PARAGRAPH.finditer(text):
        para_token = m.group(1).strip("()")
        _emit(m.group(0), f"(paragraph {para_token})")

    return out


def _split_sub_para(tail: str) -> tuple[str | None, str | None]:
    """From ``"(1)(a)"`` return ``("1", "a")``."""
    if not tail:
        return None, None
    sub: str | None = None
    para: str | None = None
    parts = re.findall(r"\(([\w]+)\)", tail)
    if parts:
        if parts[0].isdigit():
            sub = parts[0]
            if len(parts) > 1 and parts[1].isalpha():
                para = parts[1]
        elif parts[0].isalpha():
            para = parts[0]
    return sub, para
