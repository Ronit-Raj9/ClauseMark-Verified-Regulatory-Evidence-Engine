"""Word-based child chunker with overlap.

Used to split a leaf `Element.text` into small child chunks suitable for dense
embedding while keeping a deterministic mapping back to the parent's character
offsets (so the snippet round-trips through the verification gates).

Only pure-Python primitives are used here — no third-party deps — because the
chunker is on the hot path of `index_document` and is also exercised in unit
tests that must run on CPU-only test runners without loading any models.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["ChildChunk", "chunk_text"]


# A token is a maximal run of non-whitespace characters. We capture the
# character offsets of each token so chunk offsets are exact, not approximate.
_TOKEN_RE = re.compile(r"\S+")


@dataclass(frozen=True)
class ChildChunk:
    """A single child chunk: the text plus char offsets into the parent string."""

    text: str
    char_start: int
    char_end: int


def _tokens(text: str) -> list[tuple[str, int, int]]:
    """Return list of (token, start, end) over `text` using whitespace splits."""
    return [(m.group(0), m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]


def chunk_text(
    text: str,
    target_size: int,
    overlap: int,
) -> list[tuple[str, int, int]]:
    """Split `text` into roughly `target_size`-character chunks with `overlap` chars overlap.

    Splits on word boundaries (never mid-word). Returns a list of
    ``(chunk_text, char_start, char_end)`` tuples such that:

    * ``text[char_start:char_end] == chunk_text``
    * offsets are monotonically non-decreasing across chunks
    * ``char_end <= len(text)``
    * ``overlap`` is best-effort: each non-first chunk starts at the last
      token boundary at-or-before ``previous_end - overlap``
    * the empty / whitespace-only input yields an empty list
    """
    if target_size <= 0:
        msg = f"target_size must be > 0, got {target_size}"
        raise ValueError(msg)
    if overlap < 0:
        msg = f"overlap must be >= 0, got {overlap}"
        raise ValueError(msg)
    if overlap >= target_size:
        msg = f"overlap ({overlap}) must be < target_size ({target_size})"
        raise ValueError(msg)

    toks = _tokens(text)
    if not toks:
        return []

    chunks: list[tuple[str, int, int]] = []
    n = len(toks)
    i = 0
    last_start = -1

    while i < n:
        start_off = toks[i][1]
        # Grow the chunk one token at a time until adding another would exceed
        # target_size. Always keep at least one token to guarantee progress.
        j = i
        cur_end = toks[j][2]
        while j + 1 < n:
            next_end = toks[j + 1][2]
            if next_end - start_off > target_size:
                break
            j += 1
            cur_end = next_end

        chunk_text_str = text[start_off:cur_end]
        chunks.append((chunk_text_str, start_off, cur_end))

        # Guard against pathological no-progress loops (shouldn't happen because
        # the inner loop always advances `i`, but stay defensive).
        if start_off == last_start:
            break
        last_start = start_off

        if j + 1 >= n:
            break

        # Compute next start using the overlap budget. Walk backwards from j to
        # find the earliest token whose start is >= (cur_end - overlap).
        target_start_char = cur_end - overlap
        next_i = j + 1
        k = j
        while k > i and toks[k][1] >= target_start_char:
            next_i = k
            k -= 1
        # Ensure forward progress — never restart at the same token as `i`.
        if next_i <= i:
            next_i = i + 1
        i = next_i

    return chunks
