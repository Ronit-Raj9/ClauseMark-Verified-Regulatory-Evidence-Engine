"""Hybrid (BGE-M3 dense + BM25 sparse) parent-document retrieval.

Public surface:
  * `RetrievalService`     — implements `RetrievalPort`
  * `QdrantVectorStore`    — implements `VectorStorePort` (production)
  * `InMemoryVectorStore`  — implements `VectorStorePort` (tests / fakes)
  * `BgeM3Embedder`        — BGE-M3 dense + Qdrant BM25 sparse
  * `BgeReranker`          — `BAAI/bge-reranker-v2-m3` cross-encoder
  * `IdentityReranker`     — no-op reranker for tests
  * `chunk_text`           — pure-Python word-based chunker with overlap
"""

from rie_retrieval.chunker import ChildChunk, chunk_text
from rie_retrieval.embedder import (
    BGE_M3_DENSE_DIM,
    DEFAULT_DENSE_MODEL,
    DEFAULT_SPARSE_MODEL,
    BgeM3Embedder,
    DenseEmbedderPort,
    SparseEmbedderPort,
)
from rie_retrieval.fakes import (
    DummyDenseEmbedder,
    DummySparseEmbedder,
    InMemoryVectorStore,
)
from rie_retrieval.qdrant_store import QdrantVectorStore
from rie_retrieval.reranker import (
    DEFAULT_RERANKER_MODEL,
    BgeReranker,
    IdentityReranker,
)
from rie_retrieval.service import (
    DEFAULT_CHILD_CHUNK_OVERLAP,
    DEFAULT_CHILD_CHUNK_SIZE,
    DEFAULT_TOP_K_INITIAL,
    DEFAULT_TOP_K_RERANK,
    RetrievalService,
)
from rie_retrieval.tokenizer import SUPPORTED_LANGUAGES, is_supported_language, tokenize

__all__ = [
    "BGE_M3_DENSE_DIM",
    "DEFAULT_DENSE_MODEL",
    "DEFAULT_RERANKER_MODEL",
    "DEFAULT_SPARSE_MODEL",
    "DEFAULT_CHILD_CHUNK_OVERLAP",
    "DEFAULT_CHILD_CHUNK_SIZE",
    "DEFAULT_TOP_K_INITIAL",
    "DEFAULT_TOP_K_RERANK",
    "SUPPORTED_LANGUAGES",
    "BgeM3Embedder",
    "BgeReranker",
    "ChildChunk",
    "DenseEmbedderPort",
    "DummyDenseEmbedder",
    "DummySparseEmbedder",
    "IdentityReranker",
    "InMemoryVectorStore",
    "QdrantVectorStore",
    "RetrievalService",
    "SparseEmbedderPort",
    "chunk_text",
    "is_supported_language",
    "tokenize",
]
