"""RAGAS-style metrics — pure-Python proxies.

We intentionally do NOT take a hard dependency on the ``ragas`` package: it is
heavy (pulls LangChain + LLM clients), non-deterministic, and the indicator
metrics we need here are token-overlap approximations that are fast, offline,
and unit-testable. If ``ragas`` is available we expose ``RAGAS_AVAILABLE = True``
so a future caller can opt-in to the real metric; the default ``compute_*``
functions are deterministic proxies.

Proxy definitions
-----------------
- ``compute_context_recall(claim, gold_span_text, retrieved_passages)``
  Fraction of *token-set* of ``gold_span_text`` that appears in the union of
  retrieved passages. Whitespace-normalised, lowercased, punctuation-stripped
  via a conservative tokenizer. Range ``[0.0, 1.0]``. Empty gold span → 0.0.

- ``compute_faithfulness(claim_decomposition, cited_span_text)``
  Fraction of *constraint* tokens (from the ``Decomposition.constraint`` field)
  that occur in the cited span text. Captures the "is the claim supported by
  the cited evidence?" intuition without an LLM round-trip. Empty
  constraint → 0.0; identical strings → 1.0.

These are PROXIES — when a real entailment / NLI evaluator becomes available,
swap behind the same signature.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from rie_contracts import Claim, Decomposition

def _ragas_available() -> bool:
    try:  # pragma: no cover - optional dependency
        import ragas  # type: ignore[import-not-found]  # noqa: F401

        return True
    except ImportError:  # pragma: no cover
        return False


RAGAS_AVAILABLE: bool = _ragas_available()


# A conservative tokenizer: words = runs of alphanumerics (Unicode-aware via \w).
_WORD = re.compile(r"\w+", flags=re.UNICODE)


def _tokens(text: str) -> list[str]:
    """Lowercase, whitespace-normalised word tokens. Punctuation stripped."""
    return _WORD.findall(text.lower())


def _token_set(text: str) -> set[str]:
    return set(_tokens(text))


def compute_context_recall(
    claim: Claim | None,
    gold_span_text: str,
    retrieved_passages: Sequence[str],
) -> float:
    """Fraction of gold-span token-set covered by the union of retrieved passages.

    ``claim`` is accepted (and currently unused by the proxy) so a future
    RAGAS-backed implementation can correlate against the claim's clause / regime
    without changing the signature.

    Returns ``0.0`` when ``gold_span_text`` is empty (no signal to recall).
    """
    _ = claim  # reserved for the real-RAGAS path
    gold_tokens = _token_set(gold_span_text)
    if not gold_tokens:
        return 0.0
    retrieved_tokens: set[str] = set()
    for passage in retrieved_passages:
        retrieved_tokens.update(_token_set(passage))
    if not retrieved_tokens:
        return 0.0
    hits = gold_tokens & retrieved_tokens
    return len(hits) / len(gold_tokens)


def compute_faithfulness(
    claim_decomposition: Decomposition,
    cited_span_text: str,
) -> float:
    """Token-overlap proxy for faithfulness of a claim to its cited span.

    Computed as ``|constraint_tokens ∩ cited_span_tokens| / |constraint_tokens|``.

    - Empty constraint → ``0.0`` (no testable claim).
    - Identical strings → ``1.0``.
    - Disjoint vocabularies → ``0.0``.
    """
    constraint_tokens = _token_set(claim_decomposition.constraint)
    if not constraint_tokens:
        return 0.0
    cited_tokens = _token_set(cited_span_text)
    if not cited_tokens:
        return 0.0
    hits = constraint_tokens & cited_tokens
    return len(hits) / len(constraint_tokens)


def aggregate_recall(values: Iterable[float]) -> float:
    """Mean of a sequence of recall values; empty → 0.0."""
    vs = list(values)
    return sum(vs) / len(vs) if vs else 0.0


__all__ = [
    "RAGAS_AVAILABLE",
    "aggregate_recall",
    "compute_context_recall",
    "compute_faithfulness",
]
