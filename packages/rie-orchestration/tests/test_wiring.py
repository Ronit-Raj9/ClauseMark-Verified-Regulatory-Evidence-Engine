"""Unit tests for adapter wiring defaults and production-env fallback policy."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from rie_orchestration.wiring import (
    _DEFAULT_EMBEDDING_MODEL,
    _DEFAULT_RERANKER_MODEL,
    _DEFAULT_SPARSE_MODEL,
    _production_env_configured,
    build_default_bundle,
)


def test_production_env_requires_database_and_qdrant(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL_SYNC", raising=False)
    monkeypatch.delenv("QDRANT_HOST", raising=False)
    assert _production_env_configured() is False

    monkeypatch.setenv("DATABASE_URL_SYNC", "postgresql+psycopg://x")
    assert _production_env_configured() is False

    monkeypatch.setenv("QDRANT_HOST", "localhost")
    assert _production_env_configured() is True


def test_production_env_failure_does_not_fall_back_to_fakes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABASE_URL_SYNC", "postgresql+psycopg://x")
    monkeypatch.setenv("QDRANT_HOST", "localhost")
    monkeypatch.delenv("RIE_FORCE_FAKES", raising=False)

    with (
        patch(
            "rie_orchestration.wiring._build_real_bundle",
            side_effect=RuntimeError("db down"),
        ),
        pytest.raises(RuntimeError, match="db down"),
    ):
        build_default_bundle()


def test_non_production_env_falls_back_to_fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL_SYNC", raising=False)
    monkeypatch.delenv("QDRANT_HOST", raising=False)
    monkeypatch.delenv("RIE_FORCE_FAKES", raising=False)

    with patch(
        "rie_orchestration.wiring._build_real_bundle",
        side_effect=ImportError("optional dep missing"),
    ):
        bundle = build_default_bundle()

    from rie_orchestration.fakes import FakeRetrieval

    assert isinstance(bundle.retrieval, FakeRetrieval)


def test_real_bundle_uses_bge_m3_and_bge_reranker_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify model names wired into retrieval without loading heavy weights."""
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("SPARSE_MODEL", raising=False)
    monkeypatch.delenv("RERANKER_MODEL", raising=False)
    monkeypatch.setenv("DATABASE_URL_SYNC", "postgresql+psycopg://x")
    monkeypatch.setenv("QDRANT_HOST", "localhost")

    captured: dict[str, object] = {}

    class _StubEmbedder:
        def __init__(self, *, dense_model: str, sparse_model: str) -> None:
            captured["dense_model"] = dense_model
            captured["sparse_model"] = sparse_model

    class _StubReranker:
        def __init__(self, *, model_name: str) -> None:
            captured["reranker_model"] = model_name

    class _StubVectorStore:
        def __init__(self, *, host: str, port: int) -> None:
            captured["qdrant_host"] = host
            captured["qdrant_port"] = port

    class _StubRetrieval:
        def __init__(self, **kwargs: object) -> None:
            captured["vector_store"] = kwargs.get("vector_store")
            captured["reranker"] = kwargs.get("reranker")

    stub_repo = MagicMock()
    stub_repo.get_structure_edges.return_value = []
    stub_repo.get_elements.return_value = []

    with (
        patch("rie_persistence.create_engine", return_value=MagicMock()),
        patch("rie_persistence.db.make_session_factory", return_value=MagicMock()),
        patch("rie_persistence.DocumentRepository", return_value=stub_repo),
        patch("rie_retrieval.QdrantVectorStore", _StubVectorStore),
        patch("rie_retrieval.BgeM3Embedder", _StubEmbedder),
        patch("rie_retrieval.BgeReranker", _StubReranker),
        patch("rie_retrieval.RetrievalService", _StubRetrieval),
        patch("rie_classify.OllamaClient", return_value=MagicMock()),
        patch("rie_classify.ClassificationService", return_value=MagicMock()),
        patch("rie_verify.TransformersNliBackend", return_value=MagicMock()),
        patch("rie_verify.OllamaSecondLlm", return_value=MagicMock()),
        patch("rie_verify.VerificationService", return_value=MagicMock()),
        patch("rie_ingest.IngestService", return_value=MagicMock()),
        patch("rie_extract.ExtractionService", return_value=MagicMock()),
        patch("rie_coverage.CoverageReasoner", return_value=MagicMock()),
    ):
        build_default_bundle(repo_root=Path(__file__).resolve().parents[3])

    assert captured["dense_model"] == _DEFAULT_EMBEDDING_MODEL
    assert captured["sparse_model"] == _DEFAULT_SPARSE_MODEL
    assert captured["reranker_model"] == _DEFAULT_RERANKER_MODEL
    assert captured["qdrant_host"] == "localhost"
    assert captured["qdrant_port"] == 6333
    assert isinstance(captured["vector_store"], _StubVectorStore)


def test_qdrant_host_absent_uses_inmemory_vector_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABASE_URL_SYNC", "postgresql+psycopg://x")
    monkeypatch.delenv("QDRANT_HOST", raising=False)

    captured: dict[str, object] = {}

    class _StubRetrieval:
        def __init__(self, **kwargs: object) -> None:
            captured["vector_store"] = kwargs.get("vector_store")

    stub_repo = MagicMock()
    stub_repo.get_structure_edges.return_value = []
    stub_repo.get_elements.return_value = []

    with (
        patch("rie_persistence.create_engine", return_value=MagicMock()),
        patch("rie_persistence.db.make_session_factory", return_value=MagicMock()),
        patch("rie_persistence.DocumentRepository", return_value=stub_repo),
        patch("rie_retrieval.BgeM3Embedder", return_value=MagicMock()),
        patch("rie_retrieval.BgeReranker", return_value=MagicMock()),
        patch("rie_retrieval.RetrievalService", _StubRetrieval),
        patch("rie_retrieval.InMemoryVectorStore") as inmem_cls,
        patch("rie_classify.OllamaClient", return_value=MagicMock()),
        patch("rie_classify.ClassificationService", return_value=MagicMock()),
        patch("rie_verify.TransformersNliBackend", return_value=MagicMock()),
        patch("rie_verify.OllamaSecondLlm", return_value=MagicMock()),
        patch("rie_verify.VerificationService", return_value=MagicMock()),
        patch("rie_ingest.IngestService", return_value=MagicMock()),
        patch("rie_extract.ExtractionService", return_value=MagicMock()),
        patch("rie_coverage.CoverageReasoner", return_value=MagicMock()),
    ):
        build_default_bundle(repo_root=Path(__file__).resolve().parents[3])

    inmem_cls.assert_called_once()
    assert captured["vector_store"] is inmem_cls.return_value
