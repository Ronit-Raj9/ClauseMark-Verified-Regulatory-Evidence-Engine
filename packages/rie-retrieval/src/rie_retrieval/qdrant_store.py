"""Qdrant adapter for `VectorStorePort` — named-vector hybrid + RRF fusion.

Collection layout (per `tool.md` §Qdrant):
  * named dense vector ``dense``  — VectorParams(size=1024, distance=COSINE)
    sized for BGE-M3
  * named sparse vector ``sparse`` — SparseVectorParams() for BM25 / BGE-M3
    sparse

Hybrid query uses ``models.FusionQuery(fusion=models.Fusion.RRF)`` over a pair
of ``Prefetch`` clauses — one dense, one sparse — so Qdrant performs the RRF
(k=60) merge server-side and we never hand-roll fusion math in production.

Import is unconditional (``qdrant-client`` is a pinned package dep) but the
constructor only *connects* lazily on the first call to ``ensure_collection``
or ``upsert`` — instantiating the adapter never opens a TCP socket.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models
from rie_contracts import VectorStorePort

__all__ = ["QdrantVectorStore"]


_DEFAULT_DENSE_NAME = "dense"
_DEFAULT_SPARSE_NAME = "sparse"
_DEFAULT_UPSERT_BATCH = 64
_RRF_PREFETCH_LIMIT = 200  # how many candidates each leg returns before RRF.


def _to_qdrant_id(raw: str) -> str:
    """Convert an arbitrary string id to a deterministic UUID Qdrant accepts.

    Qdrant point ids must be either unsigned ints or UUIDs; our element-derived
    ids (``<doc_id>__<element_id>__c<idx>``) are neither. A namespaced UUID5
    over the raw string is stable across processes and round-trips through the
    ``raw_id`` payload field.
    """
    return str(uuid5(NAMESPACE_URL, raw))


class QdrantVectorStore(VectorStorePort):
    """Hybrid (dense + sparse) Qdrant adapter implementing `VectorStorePort`."""

    def __init__(
        self,
        client: QdrantClient | None = None,
        *,
        url: str | None = None,
        host: str = "localhost",
        port: int = 6333,
        dense_vector_name: str = _DEFAULT_DENSE_NAME,
        sparse_vector_name: str = _DEFAULT_SPARSE_NAME,
        upsert_batch: int = _DEFAULT_UPSERT_BATCH,
        prefetch_limit: int = _RRF_PREFETCH_LIMIT,
    ) -> None:
        self._client = client
        self._url = url
        self._host = host
        self._port = port
        self._dense_name = dense_vector_name
        self._sparse_name = sparse_vector_name
        self._upsert_batch = upsert_batch
        self._prefetch_limit = prefetch_limit

    # ── lazy client ──────────────────────────────────────────────────────

    def _connect(self) -> QdrantClient:
        if self._client is None:
            if self._url is not None:
                self._client = QdrantClient(url=self._url)
            else:
                self._client = QdrantClient(host=self._host, port=self._port)
        return self._client

    # ── VectorStorePort ──────────────────────────────────────────────────

    def ensure_collection(self, name: str, dense_dim: int) -> None:
        client = self._connect()
        if client.collection_exists(collection_name=name):
            return
        client.create_collection(
            collection_name=name,
            vectors_config={
                self._dense_name: models.VectorParams(
                    size=dense_dim,
                    distance=models.Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                self._sparse_name: models.SparseVectorParams(),
            },
        )

    def upsert(
        self,
        collection: str,
        ids: Sequence[str],
        dense_vectors: Sequence[Sequence[float]],
        sparse_vectors: Sequence[dict[int, float]],
        payloads: Sequence[dict[str, str | int | float | bool | None]],
    ) -> None:
        client = self._connect()
        points: list[models.PointStruct] = []
        for raw_id, dv, sv, payload in zip(
            ids, dense_vectors, sparse_vectors, payloads, strict=True
        ):
            indices = list(sv.keys())
            values = [float(sv[i]) for i in indices]
            enriched_payload: dict[str, Any] = dict(payload)
            enriched_payload["raw_id"] = raw_id
            points.append(
                models.PointStruct(
                    id=_to_qdrant_id(raw_id),
                    vector={
                        self._dense_name: list(dv),
                        self._sparse_name: models.SparseVector(
                            indices=indices,
                            values=values,
                        ),
                    },
                    payload=enriched_payload,
                )
            )

        for start in range(0, len(points), self._upsert_batch):
            batch = points[start : start + self._upsert_batch]
            client.upsert(collection_name=collection, points=batch, wait=True)

    def search_hybrid(
        self,
        collection: str,
        dense_query: Sequence[float],
        sparse_query: dict[int, float],
        top_k: int,
    ) -> list[tuple[str, float]]:
        client = self._connect()
        sparse_indices = list(sparse_query.keys())
        sparse_values = [float(sparse_query[i]) for i in sparse_indices]
        prefetch = [
            models.Prefetch(
                query=list(dense_query),
                using=self._dense_name,
                limit=self._prefetch_limit,
            ),
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse_indices,
                    values=sparse_values,
                ),
                using=self._sparse_name,
                limit=self._prefetch_limit,
            ),
        ]
        response = client.query_points(
            collection_name=collection,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )
        results: list[tuple[str, float]] = []
        for point in response.points:
            payload = point.payload or {}
            raw_id = payload.get("raw_id")
            if not isinstance(raw_id, str):
                # Fallback: stringify the Qdrant id if a caller upserted by hand.
                raw_id = str(point.id)
            results.append((raw_id, float(point.score)))
        return results
