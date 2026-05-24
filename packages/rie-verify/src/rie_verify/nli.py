"""NLI backend Protocol + concrete + fake implementations.

The NLI backend answers the *narrow* question of gate 3 (entailment): given a
premise (the cited element text) and a hypothesis (the decomposed claim
rendered as a sentence), what is the entailment probability?

This module deliberately exposes a plain Protocol — the orchestrator picks the
concrete backend. The production backend lazy-loads a cross-encoder so importing
`rie_verify` stays cheap (and test environments without `transformers` installed
are not penalised).
"""

from __future__ import annotations

from typing import Protocol, TypedDict, runtime_checkable


class NliScores(TypedDict):
    """Three-way NLI distribution. Probabilities sum to 1.0 (approximately)."""

    entailment: float
    contradiction: float
    neutral: float


@runtime_checkable
class NliBackend(Protocol):
    """A scorer that returns the three-way NLI distribution for (premise, hypothesis)."""

    def score(self, premise: str, hypothesis: str) -> NliScores: ...


class TransformersNliBackend:
    """Cross-encoder NLI via `sentence-transformers`.

    Lazy-loads `cross-encoder/nli-deberta-v3-base` on first call so importing
    this module costs nothing. The model returns logits in the order
    (contradiction, entailment, neutral); we softmax and project into
    `NliScores`.

    NOTE — this backend is *optional*. Installing `sentence-transformers` is
    delegated to the orchestrator. Tests use `FakeNliBackend` and never
    instantiate this class.
    """

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-base") -> None:
        self._model_name: str = model_name
        self._model: object | None = None

    def _ensure_loaded(self) -> object:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder  # type: ignore[import-not-found]
            except ImportError as exc:  # pragma: no cover — defensive guard
                msg = (
                    "TransformersNliBackend requires `sentence-transformers`. "
                    "Install via `uv add --package rie-verify sentence-transformers`."
                )
                raise RuntimeError(msg) from exc
            self._model = CrossEncoder(self._model_name)
        return self._model

    def score(self, premise: str, hypothesis: str) -> NliScores:
        import math

        model = self._ensure_loaded()
        # CrossEncoder returns logits over (contradiction, entailment, neutral).
        raw = model.predict([(premise, hypothesis)])  # type: ignore[attr-defined]
        # `raw` is an ndarray of shape (1, 3); convert to plain Python.
        row = list(raw[0])  # type: ignore[index]
        contradiction_logit = float(row[0])
        entailment_logit = float(row[1])
        neutral_logit = float(row[2])

        # Softmax for honest probabilities.
        max_logit = max(contradiction_logit, entailment_logit, neutral_logit)
        exps = [
            math.exp(contradiction_logit - max_logit),
            math.exp(entailment_logit - max_logit),
            math.exp(neutral_logit - max_logit),
        ]
        z = sum(exps)
        return NliScores(
            contradiction=exps[0] / z,
            entailment=exps[1] / z,
            neutral=exps[2] / z,
        )


class FakeNliBackend:
    """Deterministic NLI backend for tests — always returns the configured scores."""

    def __init__(
        self,
        entailment_prob: float = 0.9,
        contradiction_prob: float = 0.05,
        neutral_prob: float = 0.05,
    ) -> None:
        self._scores: NliScores = NliScores(
            entailment=entailment_prob,
            contradiction=contradiction_prob,
            neutral=neutral_prob,
        )

    def score(self, premise: str, hypothesis: str) -> NliScores:
        return self._scores
