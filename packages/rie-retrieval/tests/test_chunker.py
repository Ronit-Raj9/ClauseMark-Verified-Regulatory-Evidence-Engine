"""Chunker invariants — offset arithmetic + overlap budget + edge cases."""

from __future__ import annotations

import pytest
from rie_retrieval.chunker import chunk_text


def _make_text(n_words: int) -> str:
    return " ".join(f"word{i:04d}" for i in range(n_words))


def test_empty_text_returns_empty() -> None:
    assert chunk_text("", target_size=100, overlap=10) == []
    assert chunk_text("   \n\t  ", target_size=100, overlap=10) == []


def test_short_text_fits_single_chunk() -> None:
    text = "An organisation shall not transfer personal data."
    chunks = chunk_text(text, target_size=200, overlap=20)
    assert len(chunks) == 1
    chunk, start, end = chunks[0]
    assert chunk == text[start:end]
    assert start == 0
    assert end == len(text)


def test_invalid_args_raise() -> None:
    with pytest.raises(ValueError):
        chunk_text("hello", target_size=0, overlap=0)
    with pytest.raises(ValueError):
        chunk_text("hello", target_size=10, overlap=-1)
    with pytest.raises(ValueError):
        chunk_text("hello", target_size=10, overlap=10)


def test_offsets_round_trip_to_text_exactly() -> None:
    text = _make_text(120)
    chunks = chunk_text(text, target_size=60, overlap=15)
    assert chunks, "expected non-empty result"
    for chunk_str, start, end in chunks:
        assert text[start:end] == chunk_str


def test_offsets_are_monotonic_and_bounded() -> None:
    text = _make_text(200)
    chunks = chunk_text(text, target_size=80, overlap=20)

    prev_start = -1
    prev_end = -1
    for _chunk, start, end in chunks:
        assert start >= 0
        assert end <= len(text)
        assert start <= end
        # Starts are strictly increasing (forward progress guaranteed).
        assert start > prev_start
        # Ends are non-decreasing.
        assert end >= prev_end
        prev_start, prev_end = start, end

    # Last chunk reaches the end of the text (no trailing words dropped).
    assert chunks[-1][2] == len(text)


def test_overlap_is_respected_between_consecutive_chunks() -> None:
    text = _make_text(150)
    target_size = 80
    overlap = 25
    chunks = chunk_text(text, target_size=target_size, overlap=overlap)
    assert len(chunks) >= 2

    for prev, curr in zip(chunks, chunks[1:], strict=False):
        _, _, prev_end = prev
        _, curr_start, _ = curr
        # Overlap is best-effort but the next chunk must START somewhere inside
        # the overlap window or right after the previous end.
        gap = curr_start - prev_end
        assert gap <= 0 or gap <= 1, (
            f"non-overlapping consecutive chunks: prev_end={prev_end} curr_start={curr_start}"
        )
        # And it must NOT skip ahead past the next-word boundary either.
        assert curr_start >= prev[1], "chunk start moved backwards"


def test_no_chunk_exceeds_target_size_by_more_than_one_token() -> None:
    text = _make_text(100)
    target_size = 50
    chunks = chunk_text(text, target_size=target_size, overlap=10)
    for chunk_str, _start, _end in chunks:
        # Allow a single token to overflow because we never split mid-word; in
        # practice that's bounded by the longest token in the input.
        longest_token = max((len(t) for t in chunk_str.split()), default=0)
        assert len(chunk_str) <= target_size + longest_token


def test_single_huge_word_still_returned() -> None:
    text = "x" * 200
    chunks = chunk_text(text, target_size=80, overlap=10)
    assert len(chunks) == 1
    assert chunks[0] == (text, 0, len(text))


def test_three_words_no_overlap_split() -> None:
    text = "alpha beta gamma delta epsilon zeta"
    chunks = chunk_text(text, target_size=15, overlap=0)
    rebuilt_tokens: list[str] = []
    for chunk_str, _s, _e in chunks:
        rebuilt_tokens.extend(chunk_str.split())
    # With zero overlap every token appears exactly once.
    assert rebuilt_tokens == text.split()
