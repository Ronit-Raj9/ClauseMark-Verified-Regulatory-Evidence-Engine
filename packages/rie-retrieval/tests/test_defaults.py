"""Verify §6.1 architecture default model names."""

from __future__ import annotations

import inspect

from rie_retrieval import (
    DEFAULT_DENSE_MODEL,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_SPARSE_MODEL,
    BgeM3Embedder,
    BgeReranker,
)


def test_embedder_constructor_defaults_match_architecture() -> None:
    sig = inspect.signature(BgeM3Embedder.__init__)
    assert sig.parameters["dense_model"].default == DEFAULT_DENSE_MODEL
    assert sig.parameters["sparse_model"].default == DEFAULT_SPARSE_MODEL
    assert DEFAULT_DENSE_MODEL == "BAAI/bge-m3"
    assert DEFAULT_SPARSE_MODEL == "Qdrant/bm25"


def test_reranker_constructor_default_matches_architecture() -> None:
    sig = inspect.signature(BgeReranker.__init__)
    assert sig.parameters["model_name"].default == DEFAULT_RERANKER_MODEL
    assert DEFAULT_RERANKER_MODEL == "BAAI/bge-reranker-v2-m3"
