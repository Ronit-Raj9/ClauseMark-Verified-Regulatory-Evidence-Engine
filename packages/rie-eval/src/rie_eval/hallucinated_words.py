"""Hallucinated-words metric.

Definition
----------
``compute_hallucinated_words_rate(extracted_text, source_text)`` returns the
fraction of *word tokens* in ``extracted_text`` that do NOT appear in
``source_text`` (case-insensitive, whitespace-normalised, punctuation
stripped). 0.0 means every extracted word was present in the source; 1.0 means
none were.

This is the cheapest possible "did the model invent words?" signal and is a
useful canary for downstream OCR / VLM hallucinations. It is intentionally
*word-bag*-shaped — order, syntax, and semantics are explicitly ignored.
"""

from __future__ import annotations

import re

_WORD = re.compile(r"\w+", flags=re.UNICODE)


def _normalise(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace → word tokens."""
    return _WORD.findall(text.lower())


def compute_hallucinated_words_rate(extracted_text: str, source_text: str) -> float:
    """Fraction of words in ``extracted_text`` absent from ``source_text``.

    - Empty ``extracted_text`` → ``0.0`` (nothing to hallucinate).
    - Identical strings → ``0.0``.
    - Single foreign word in N-word extraction → ``1/N``.
    """
    extracted_tokens = _normalise(extracted_text)
    if not extracted_tokens:
        return 0.0
    source_set = set(_normalise(source_text))
    hallucinated = sum(1 for t in extracted_tokens if t not in source_set)
    return hallucinated / len(extracted_tokens)


__all__ = ["compute_hallucinated_words_rate"]
