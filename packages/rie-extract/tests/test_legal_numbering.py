"""Unit tests for the legal-numbering grammar."""

from __future__ import annotations

from rie_extract.structure.legal_numbering import (
    find_cross_references,
    parse_numbering,
)


def test_parse_section_heading() -> None:
    assert parse_numbering("Section 26.") == "s. 26"
    assert parse_numbering("Section 26. Transfer of personal data") == "s. 26"
    assert parse_numbering("Sec. 5") == "s. 5"
    assert parse_numbering("s. 12A") == "s. 12A"


def test_parse_article_heading() -> None:
    assert parse_numbering("Article 12") == "art. 12"
    assert parse_numbering("Art. 12(1)") == "art. 12(1)"
    assert parse_numbering("ART 4") == "art. 4"


def test_parse_chapter_heading() -> None:
    assert parse_numbering("CHAPTER IV — CROSS-BORDER TRANSFER") == "ch. IV"
    assert parse_numbering("Chapter 2") == "ch. 2"


def test_parse_non_heading_returns_none() -> None:
    assert parse_numbering("An organisation shall not transfer") is None
    assert parse_numbering("") is None


def test_find_cross_references_section() -> None:
    text = "as specified in section 6"
    refs = find_cross_references(text)
    assert ("section 6", "s. 6") in refs


def test_find_cross_references_compound() -> None:
    text = "for the purposes of subsection (1) of section 26"
    refs = find_cross_references(text)
    canonicals = {c for _, c in refs}
    assert "s. 26" in canonicals
    assert "(subsection (1))" in canonicals


def test_find_cross_references_article() -> None:
    text = "in accordance with Article 26(1)(a)"
    refs = find_cross_references(text)
    canonicals = {c for _, c in refs}
    assert "art. 26(1)(a)" in canonicals


def test_find_cross_references_paragraph() -> None:
    text = "as set out in paragraph (a)"
    refs = find_cross_references(text)
    canonicals = {c for _, c in refs}
    assert "(paragraph a)" in canonicals


def test_find_cross_references_deduplicates() -> None:
    text = "section 12 and again section 12 and section 12"
    refs = find_cross_references(text)
    canonicals = [c for _, c in refs]
    assert canonicals.count("s. 12") == 1
