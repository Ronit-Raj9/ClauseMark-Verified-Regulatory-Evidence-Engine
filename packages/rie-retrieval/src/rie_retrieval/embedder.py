"""BGE-M3 dense + Qdrant BM25 sparse embedders (lazy-loaded).

Per `tool.md` (§Qdrant + BGE-M3): the sensible default for legal text is the
BGE-M3 dense vector for semantics combined with Qdrant's BM25 sparse vector for
exact-term recall (section numbers, defined terms). The two are wired
explicitly here rather than letting a wrapper guess.

Models are loaded on first use so that:
  * importing this module never pulls multi-hundred-MB ONNX weights from disk
  * CPU-only test runners can import-and-skip without ever instantiating the
    real embedder
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

__all__ = [
    "BGE_M3_DENSE_DIM",
    "DEFAULT_DENSE_MODEL",
    "DEFAULT_SPARSE_MODEL",
    "BgeM3Embedder",
    "DenseEmbedderPort",
    "SparseEmbedderPort",
]


# BGE-M3 dense vector dimensionality (fixed by the model architecture).
BGE_M3_DENSE_DIM: int = 1024

# §6.1 architecture defaults — single source of truth for wiring / env fallbacks.
DEFAULT_DENSE_MODEL: str = "BAAI/bge-m3"
DEFAULT_SPARSE_MODEL: str = "Qdrant/bm25"


@runtime_checkable
class DenseEmbedderPort(Protocol):
    """Anything that turns texts into fixed-width dense vectors.

    `lang` is optional and accepted by the production BGE-M3 embedder for
    parity with the sparse leg; it is intentionally a hint, not a routing key
    — BGE-M3 is multilingual and a missing `lang` is never an error.
    """

    def embed_dense(
        self, texts: Sequence[str], lang: str | None = None
    ) -> list[list[float]]: ...


@runtime_checkable
class SparseEmbedderPort(Protocol):
    """Anything that turns texts into BM25-style sparse term-id -> weight maps.

    `lang` drives per-language tokenization (NFKC normalisation, FR-style
    elision split, CJK bigram fallback, stopword set). A missing `lang`
    falls back to English-style behaviour — the historical default.
    """

    def embed_sparse(
        self, texts: Sequence[str], lang: str | None = None
    ) -> list[dict[int, float]]: ...


class BgeM3Embedder:
    """Production embedder: BGE-M3 dense + Qdrant BM25 sparse via `fastembed`.

    Both inner models are lazy-loaded singletons keyed on the constructor
    arguments — instantiating this class is free; loading happens on the first
    call to ``embed_dense`` / ``embed_sparse``.
    """

    _dense_singletons: dict[str, object] = {}
    _sparse_singletons: dict[str, object] = {}

    def __init__(
        self,
        dense_model: str = DEFAULT_DENSE_MODEL,
        sparse_model: str = DEFAULT_SPARSE_MODEL,
    ) -> None:
        self._dense_model_name = dense_model
        self._sparse_model_name = sparse_model

    # ── lazy accessors ────────────────────────────────────────────────────

    def _dense(self) -> object:
        cached = self._dense_singletons.get(self._dense_model_name)
        if cached is None:
            from fastembed import TextEmbedding  # noqa: PLC0415 — lazy import

            cached = TextEmbedding(model_name=self._dense_model_name)
            self._dense_singletons[self._dense_model_name] = cached
        return cached

    def _sparse(self) -> object:
        cached = self._sparse_singletons.get(self._sparse_model_name)
        if cached is None:
            from fastembed import SparseTextEmbedding  # noqa: PLC0415 — lazy import

            cached = SparseTextEmbedding(model_name=self._sparse_model_name)
            self._sparse_singletons[self._sparse_model_name] = cached
        return cached

    # ── public API ────────────────────────────────────────────────────────

    def embed_dense(
        self, texts: Sequence[str], lang: str | None = None
    ) -> list[list[float]]:
        # `lang` is accepted for API parity but unused: BGE-M3 is multilingual
        # and does not take a language hint at inference time.
        del lang
        if not texts:
            return []
        model = self._dense()
        out: list[list[float]] = []
        for vec in model.embed(list(texts)):  # type: ignore[attr-defined]
            # fastembed returns numpy arrays; coerce to plain Python lists so
            # downstream serialisers (qdrant-client / json) don't need numpy.
            out.append([float(x) for x in vec])
        return out

    def embed_sparse(
        self, texts: Sequence[str], lang: str | None = None
    ) -> list[dict[int, float]]:
        # `lang` is accepted for API parity. The Qdrant/bm25 sparse model used
        # here does its own tokenization, so this hint is recorded for caller
        # observability but not forwarded — when we swap to a per-language
        # BM25 vocabulary we'll route on `lang` here.
        del lang
        if not texts:
            return []
        model = self._sparse()
        out: list[dict[int, float]] = []
        for sv in model.embed(list(texts)):  # type: ignore[attr-defined]
            # fastembed SparseEmbedding exposes `.indices` and `.values` arrays.
            indices = sv.indices  # type: ignore[attr-defined]
            values = sv.values  # type: ignore[attr-defined]
            out.append({int(i): float(v) for i, v in zip(indices, values, strict=True)})
        return out
