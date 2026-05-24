"""Cross-encoder reranker (lazy-loaded) + an identity reranker for tests.

§6.1 of `systemArchitecture.md`: reranking is the highest-leverage precision
lever and is *never* skipped. The default model is the FastEmbed-hosted
`Xenova/ms-marco-MiniLM-L-6-v2` cross-encoder — small enough to run on CPU,
strong enough to materially improve MRR@5 on top of hybrid retrieval.

PRODUCTION MULTILINGUAL NOTE (Phase 2):
  `Xenova/ms-marco-MiniLM-L-6-v2` is trained on English MS-MARCO and will
  silently under-rank CJK and Arabic candidates. For multilingual deployments
  swap `model_name` to `BAAI/bge-reranker-v2-m3` (or `jinaai/jina-reranker-v2-
  base-multilingual`) — both cover en/fr/es/de/zh/ar with comparable latency.
  The constructor accepts an explicit `model_name` so the swap is a single
  argument at wiring time, no code change inside this module.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = ["BgeReranker", "IdentityReranker"]


class BgeReranker:
    """Production reranker — wraps a FastEmbed cross-encoder.

    The model is lazy-loaded on first call so importing this module is free.
    """

    _singletons: dict[str, object] = {}

    def __init__(self, model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2") -> None:
        self._model_name = model_name

    def _model(self) -> object:
        cached = self._singletons.get(self._model_name)
        if cached is None:
            from fastembed.rerank.cross_encoder import (  # noqa: PLC0415 — lazy
                TextCrossEncoder,
            )

            cached = TextCrossEncoder(model_name=self._model_name)
            self._singletons[self._model_name] = cached
        return cached

    def rerank(self, query: str, candidates: Sequence[tuple[str, str]]) -> list[tuple[str, float]]:
        """Return `(id, score)` sorted by descending cross-encoder score."""
        if not candidates:
            return []
        model = self._model()
        documents = [text for _, text in candidates]
        scores = list(model.rerank(query, documents))  # type: ignore[attr-defined]
        scored: list[tuple[str, float]] = [
            (cand_id, float(score)) for (cand_id, _), score in zip(candidates, scores, strict=True)
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored


class IdentityReranker:
    """No-op reranker for tests.

    Returns candidates in their input order with a score of 1.0, 0.99, 0.98, …
    so consumers can still treat the output as sorted-by-descending-score.
    """

    def rerank(self, query: str, candidates: Sequence[tuple[str, str]]) -> list[tuple[str, float]]:
        return [(cid, 1.0 - 0.01 * idx) for idx, (cid, _) in enumerate(candidates)]
