"""Tests for ``rie_ui.highlight.render_with_highlight``."""

from __future__ import annotations

import pytest
from rie_ui.highlight import render_with_highlight


def test_no_spans_returns_escaped_text() -> None:
    out = render_with_highlight("<b>hi</b> & bye", [])
    assert out == "&lt;b&gt;hi&lt;/b&gt; &amp; bye"
    assert "<mark>" not in out


def test_single_span_wraps_correctly() -> None:
    text = "hello world"
    out = render_with_highlight(text, [(6, 11)])
    assert out == "hello <mark>world</mark>"


def test_multiple_disjoint_spans() -> None:
    text = "abcdefghij"
    out = render_with_highlight(text, [(0, 2), (5, 7)])
    assert out == "<mark>ab</mark>cde<mark>fg</mark>hij"


def test_overlapping_spans_are_merged() -> None:
    text = "abcdefghij"
    out = render_with_highlight(text, [(1, 5), (3, 7)])
    # Merged into a single (1, 7).
    assert out == "a<mark>bcdefg</mark>hij"
    assert out.count("<mark>") == 1
    assert out.count("</mark>") == 1


def test_adjacent_spans_are_merged() -> None:
    text = "abcdef"
    # (1, 3) touches (3, 5) — merged into (1, 5).
    out = render_with_highlight(text, [(1, 3), (3, 5)])
    assert out == "a<mark>bcde</mark>f"
    assert out.count("<mark>") == 1


def test_unsorted_input_is_sorted() -> None:
    text = "abcdefghij"
    out = render_with_highlight(text, [(7, 9), (1, 3)])
    # text[1:3]="bc", text[7:9]="hi"; spans end-exclusive.
    assert out == "a<mark>bc</mark>defg<mark>hi</mark>j"


def test_html_in_text_is_escaped_outside_marks() -> None:
    text = 'a<b>"c"</b>d'
    out = render_with_highlight(text, [(0, 1)])
    assert out.startswith("<mark>a</mark>")
    # Quotes + tag chars all escaped.
    assert "&lt;b&gt;" in out
    assert "&quot;c&quot;" in out
    assert "<b>" not in out  # raw HTML must NOT survive


def test_html_inside_span_is_also_escaped() -> None:
    text = "<script>"
    out = render_with_highlight(text, [(0, len(text))])
    assert out == "<mark>&lt;script&gt;</mark>"


def test_empty_span_is_dropped() -> None:
    text = "abcdef"
    out = render_with_highlight(text, [(2, 2)])
    assert out == "abcdef"
    assert "<mark>" not in out


def test_span_past_end_is_clamped() -> None:
    text = "abc"
    out = render_with_highlight(text, [(1, 100)])
    assert out == "a<mark>bc</mark>"


def test_negative_indices_are_clamped() -> None:
    text = "abcdef"
    out = render_with_highlight(text, [(-5, 3)])
    assert out == "<mark>abc</mark>def"


def test_negative_start_and_end_drop() -> None:
    text = "abc"
    out = render_with_highlight(text, [(-10, -5)])
    assert out == "abc"


def test_reversed_span_raises() -> None:
    with pytest.raises(ValueError):
        render_with_highlight("hello", [(4, 2)])


def test_non_tuple_span_raises_type_error() -> None:
    with pytest.raises(TypeError):
        render_with_highlight("hello", ["nope"])  # type: ignore[list-item]


def test_three_way_overlap_merged_into_one() -> None:
    text = "abcdefghij"
    out = render_with_highlight(text, [(0, 4), (2, 6), (5, 9)])
    assert out == "<mark>abcdefghi</mark>j"
    assert out.count("<mark>") == 1


def test_preserves_newlines() -> None:
    text = "line1\nline2"
    out = render_with_highlight(text, [(0, 5)])
    assert "\n" in out
    assert out == "<mark>line1</mark>\nline2"


def test_doc_text_must_be_str() -> None:
    with pytest.raises(TypeError):
        render_with_highlight(123, [(0, 1)])  # type: ignore[arg-type]
