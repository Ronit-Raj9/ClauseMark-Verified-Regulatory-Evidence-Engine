"""Adapter wiring — constructs concrete adapter instances behind ports.

This is the ONE module that imports concrete adapter classes. Every other
orchestration module sees only the `Protocol` ports from `rie_contracts`.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from rie_contracts import (
    ClassifierPort,
    ConfigRepositoryPort,
    CoverageReasonerPort,
    DocumentExtractorPort,
    DocumentRepositoryPort,
    IngestPort,
    RetrievalPort,
    VerifierPort,
)

log = logging.getLogger(__name__)


@dataclass
class AdapterBundle:
    config: ConfigRepositoryPort
    ingest: IngestPort
    extractor: DocumentExtractorPort
    retrieval: RetrievalPort
    classifier: ClassifierPort
    verifier: VerifierPort
    coverage: CoverageReasonerPort
    repo: DocumentRepositoryPort
    samples_dir: Path


def build_default_bundle(
    repo_root: Path | None = None,
    *,
    use_fakes: bool = False,
) -> AdapterBundle:
    """Wire concrete adapters from each rie-* package.

    `use_fakes=True` (or RIE_FORCE_FAKES=1 env) forces the in-memory bundle —
    useful for tests, dry runs, and the demo when no Qdrant/Ollama running.
    """
    repo_root = repo_root or Path(__file__).resolve().parents[4]
    samples_dir = repo_root / "data" / "samples"

    from rie_config import ConfigRepository

    config = ConfigRepository(repo_root=repo_root)

    if use_fakes or os.getenv("RIE_FORCE_FAKES") == "1":
        return _build_fake_bundle(config, samples_dir)

    try:
        return _build_real_bundle(config, samples_dir)
    except (ImportError, RuntimeError, OSError) as e:
        log.warning("real bundle wiring failed (%s) — falling back to fakes", e)
        return _build_fake_bundle(config, samples_dir)


def _build_real_bundle(config: ConfigRepositoryPort, samples_dir: Path) -> AdapterBundle:
    """Concrete production adapters wired from env config."""
    from rie_classify import ClassificationService, OllamaClient
    from rie_coverage import CoverageReasoner
    from rie_extract import ExtractionService
    from rie_ingest import IngestService
    from rie_persistence import DocumentRepository, create_engine
    from rie_persistence.db import make_session_factory
    from rie_retrieval import (
        BgeM3Embedder,
        BgeReranker,
        InMemoryVectorStore,
        QdrantVectorStore,
        RetrievalService,
    )
    from rie_verify import OllamaSecondLlm, TransformersNliBackend, VerificationService

    engine = create_engine(os.getenv("DATABASE_URL_SYNC"))
    session_factory = make_session_factory(engine)
    repo = DocumentRepository(session_factory=session_factory)

    vector_store: object
    if os.getenv("QDRANT_HOST"):
        vector_store = QdrantVectorStore(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", "6333")),
        )
    else:
        vector_store = InMemoryVectorStore()

    embedder = BgeM3Embedder(
        dense_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5"),
        sparse_model=os.getenv("SPARSE_MODEL", "Qdrant/bm25"),
    )
    retrieval = RetrievalService(
        vector_store=vector_store,
        reranker=BgeReranker(),
        dense_embedder=embedder,
        sparse_embedder=embedder,
        collection=os.getenv("QDRANT_COLLECTION", "rie_clauses"),
    )

    llm = OllamaClient(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        model=os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_K_M"),
    )
    classifier = ClassificationService(llm=llm, get_element=repo.get_element)

    verifier = VerificationService(
        nli_model=TransformersNliBackend(
            model_name=os.getenv("NLI_MODEL", "cross-encoder/nli-deberta-v3-base")
        ),
        second_llm=OllamaSecondLlm(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            model=os.getenv("OLLAMA_VERIFIER_MODEL", "mistral:7b-instruct-q4_K_M"),
        ),
    )

    repo_root_path = samples_dir.parent.parent  # data/samples → repo root
    return AdapterBundle(
        config=config,
        ingest=IngestService(repo_root=repo_root_path, config=config),
        extractor=ExtractionService(),
        retrieval=retrieval,
        classifier=classifier,
        verifier=verifier,
        coverage=CoverageReasoner(),
        repo=repo,
        samples_dir=samples_dir,
    )


def _build_fake_bundle(config: ConfigRepositoryPort, samples_dir: Path) -> AdapterBundle:
    """In-memory adapter bundle for dry-runs + tests."""
    from rie_orchestration.fakes import (
        FakeClassifier,
        FakeCoverage,
        FakeExtractor,
        FakeIngest,
        FakeRepo,
        FakeRetrieval,
        FakeVerifier,
    )

    repo = FakeRepo()
    return AdapterBundle(
        config=config,
        ingest=FakeIngest(config),
        extractor=FakeExtractor(),
        retrieval=FakeRetrieval(),
        classifier=FakeClassifier(get_element=repo.get_element),
        verifier=FakeVerifier(),
        coverage=FakeCoverage(),
        repo=repo,
        samples_dir=samples_dir,
    )
