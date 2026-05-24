"""Smoke tests for the in-memory `VectorStorePort` fake.

The contract under test isn't "exact ranking" — it's:
  * upsert + search round-trip preserves ids
  * dense and sparse legs are both consulted (RRF merges by rank, so a doc
    that's strong on either leg surfaces)
  * jurisdiction-style payload survives upsert and is recoverable
  * `runtime_checkable` Protocol claims hold
"""

from __future__ import annotations

from rie_contracts import VectorStorePort
from rie_retrieval.fakes import (
    DummyDenseEmbedder,
    DummySparseEmbedder,
    InMemoryVectorStore,
)

COLLECTION = "test_col"


def _embed(texts: list[str]) -> tuple[list[list[float]], list[dict[int, float]]]:
    return DummyDenseEmbedder().embed_dense(texts), DummySparseEmbedder().embed_sparse(texts)


def test_in_memory_store_implements_port() -> None:
    store = InMemoryVectorStore()
    assert isinstance(store, VectorStorePort)


def test_ensure_collection_is_idempotent() -> None:
    store = InMemoryVectorStore()
    store.ensure_collection(COLLECTION, dense_dim=64)
    store.ensure_collection(COLLECTION, dense_dim=64)  # second call is a no-op
    assert store.size(COLLECTION) == 0


def test_upsert_then_hybrid_search_returns_relevant_doc_first() -> None:
    store = InMemoryVectorStore()
    store.ensure_collection(COLLECTION, dense_dim=64)

    docs = [
        ("d_cross_border", "outbound transfer comparable protection cross-border"),
        ("d_definition", "personal data means any information"),
        ("d_consent", "consent shall be freely given and specific"),
        ("d_breach", "data breach notification within seventy-two hours"),
    ]
    ids = [d[0] for d in docs]
    texts = [d[1] for d in docs]
    dense, sparse = _embed(texts)
    payloads: list[dict[str, str | int | float | bool | None]] = [
        {"jurisdiction": "SAMPLE", "snippet": t} for t in texts
    ]
    store.upsert(COLLECTION, ids, dense, sparse, payloads)

    assert store.size(COLLECTION) == 4

    query = "outbound transfer comparable protection"
    q_dense = DummyDenseEmbedder().embed_dense([query])[0]
    q_sparse = DummySparseEmbedder().embed_sparse([query])[0]

    hits = store.search_hybrid(COLLECTION, q_dense, q_sparse, top_k=4)
    assert len(hits) == 4
    top_id, top_score = hits[0]
    assert top_id == "d_cross_border"
    assert top_score > 0


def test_search_respects_top_k() -> None:
    store = InMemoryVectorStore()
    store.ensure_collection(COLLECTION, dense_dim=64)
    ids = [f"doc_{i}" for i in range(6)]
    texts = [f"clause number {i} about transfer obligations" for i in range(6)]
    dense, sparse = _embed(texts)
    payloads: list[dict[str, str | int | float | bool | None]] = [{}] * 6
    store.upsert(COLLECTION, ids, dense, sparse, payloads)

    q_dense = DummyDenseEmbedder().embed_dense(["transfer obligations"])[0]
    q_sparse = DummySparseEmbedder().embed_sparse(["transfer obligations"])[0]

    hits = store.search_hybrid(COLLECTION, q_dense, q_sparse, top_k=2)
    assert len(hits) == 2


def test_payload_round_trips() -> None:
    store = InMemoryVectorStore()
    store.ensure_collection(COLLECTION, dense_dim=64)
    dense, sparse = _embed(["hello world"])
    store.upsert(
        COLLECTION,
        ["only"],
        dense,
        sparse,
        [{"jurisdiction": "SAMPLE", "char_start": 5, "snippet": "hello world"}],
    )
    payload = store.get_payload(COLLECTION, "only")
    assert payload["jurisdiction"] == "SAMPLE"
    assert payload["char_start"] == 5
    assert payload["snippet"] == "hello world"


def test_search_on_empty_collection_returns_empty() -> None:
    store = InMemoryVectorStore()
    store.ensure_collection(COLLECTION, dense_dim=64)
    q_dense = DummyDenseEmbedder().embed_dense(["anything"])[0]
    q_sparse = DummySparseEmbedder().embed_sparse(["anything"])[0]
    assert store.search_hybrid(COLLECTION, q_dense, q_sparse, top_k=5) == []
