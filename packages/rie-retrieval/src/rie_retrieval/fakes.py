"""Deterministic in-memory fakes for the retrieval ports.

These live in the production package (not the test tree) so:
  * orchestration / contract tests in other packages can build a fully wired
    retrieval pipeline without spinning up Qdrant or downloading models
  * unit tests in this package can exercise `RetrievalService` end-to-end on
    CPU-only runners in <1 s
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from collections.abc import Sequence

from rie_contracts import VectorStorePort

__all__ = [
    "DummyDenseEmbedder",
    "DummySparseEmbedder",
    "IdentityReranker",
    "InMemoryVectorStore",
]

# Re-export the test reranker from the production module so a single import
# (`from rie_retrieval.fakes import IdentityReranker`) is the test entry point.
from rie_retrieval.reranker import IdentityReranker

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_DENSE_DIM = 64


def _tokenise(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class DummyDenseEmbedder:
    """Hash-bucketed bag-of-words → cosine-friendly unit vectors.

    Two texts that share many tokens will sit close in the resulting vector
    space; two unrelated texts will be near-orthogonal. Deterministic, fast,
    and dependency-free.
    """

    dim: int = _DENSE_DIM

    def embed_dense(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for tok in _tokenise(text):
                h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
                bucket = h % self.dim
                # Sign bit on a separate byte so collisions don't always add.
                sign = 1.0 if ((h >> 16) & 1) == 0 else -1.0
                vec[bucket] += sign
            norm = math.sqrt(sum(x * x for x in vec))
            if norm > 0:
                vec = [x / norm for x in vec]
            out.append(vec)
        return out


class DummySparseEmbedder:
    """Token-id → frequency BM25-ish encoder using stable token hashing."""

    def embed_sparse(self, texts: Sequence[str]) -> list[dict[int, float]]:
        out: list[dict[int, float]] = []
        for text in texts:
            freq: dict[int, float] = defaultdict(float)
            for tok in _tokenise(text):
                h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
                # Restrict to a 24-bit id space — plenty for tests, small for printing.
                tid = h % (1 << 24)
                freq[tid] += 1.0
            out.append(dict(freq))
        return out


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    return float(sum(x * y for x, y in zip(a, b, strict=True)))


def _sparse_dot(a: dict[int, float], b: dict[int, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return float(sum(v * b.get(k, 0.0) for k, v in a.items()))


def _rrf_fuse(
    ranked_lists: Sequence[Sequence[tuple[str, float]]], k: int = 60
) -> list[tuple[str, float]]:
    """Reciprocal-rank-fusion the way Qdrant documents it (k=60)."""
    scores: dict[str, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, (pid, _score) in enumerate(ranked, start=1):
            scores[pid] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


class InMemoryVectorStore(VectorStorePort):
    """Tiny in-process VectorStorePort — sufficient for hybrid-retrieval tests.

    Dense scoring is cosine similarity (vectors are stored as-is; the dummy
    embedder normalises them). Sparse scoring is the dot product over shared
    term ids. Fusion is RRF with k=60 — same shape as Qdrant's `FusionQuery`.
    """

    def __init__(self) -> None:
        self._collections: dict[
            str,
            dict[
                str,
                tuple[list[float], dict[int, float], dict[str, str | int | float | bool | None]],
            ],
        ] = {}

    # ── VectorStorePort ──────────────────────────────────────────────────

    def ensure_collection(self, name: str, dense_dim: int) -> None:
        self._collections.setdefault(name, {})

    def upsert(
        self,
        collection: str,
        ids: Sequence[str],
        dense_vectors: Sequence[Sequence[float]],
        sparse_vectors: Sequence[dict[int, float]],
        payloads: Sequence[dict[str, str | int | float | bool | None]],
    ) -> None:
        store = self._collections.setdefault(collection, {})
        for pid, dv, sv, payload in zip(ids, dense_vectors, sparse_vectors, payloads, strict=True):
            store[pid] = (list(dv), dict(sv), dict(payload))

    def search_hybrid(
        self,
        collection: str,
        dense_query: Sequence[float],
        sparse_query: dict[int, float],
        top_k: int,
    ) -> list[tuple[str, float]]:
        store = self._collections.get(collection, {})
        if not store:
            return []
        dense_ranked = sorted(
            ((pid, _cosine(dense_query, dv)) for pid, (dv, _sv, _p) in store.items()),
            key=lambda kv: kv[1],
            reverse=True,
        )
        sparse_ranked = sorted(
            ((pid, _sparse_dot(sparse_query, sv)) for pid, (_dv, sv, _p) in store.items()),
            key=lambda kv: kv[1],
            reverse=True,
        )
        fused = _rrf_fuse([dense_ranked, sparse_ranked])
        return fused[:top_k]

    # ── test helpers (NOT on the port) ───────────────────────────────────

    def get_payload(self, collection: str, pid: str) -> dict[str, str | int | float | bool | None]:
        return dict(self._collections[collection][pid][2])

    def size(self, collection: str) -> int:
        return len(self._collections.get(collection, {}))
