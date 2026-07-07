"""Adapter wiring — constructs concrete adapter instances behind ports.

This is the ONE module that imports concrete adapter classes. Every other
orchestration module sees only the `Protocol` ports from `rie_contracts`.

Production env (``build_default_bundle`` uses real adapters, never fakes):
  * ``DATABASE_URL_SYNC`` — Postgres for ``DocumentRepository``
  * ``QDRANT_HOST`` — enables ``QdrantVectorStore`` + ``RetrievalService``
  * ``OLLAMA_BASE_URL`` — classifier + verifier LLM (default localhost:11434)
  * ``EMBEDDING_MODEL`` — dense leg (default ``BAAI/bge-m3``)
  * ``SPARSE_MODEL`` — BM25 sparse leg (default ``Qdrant/bm25``)
  * ``RERANKER_MODEL`` — cross-encoder (default ``BAAI/bge-reranker-v2-m3``)
  * ``NLI_MODEL`` — entailment gate (default ``cross-encoder/nli-deberta-v3-base``)

Optional overrides: ``QDRANT_PORT``, ``QDRANT_COLLECTION``, ``OLLAMA_MODEL``,
``OLLAMA_VERIFIER_MODEL``.

Set ``RIE_FORCE_FAKES=1`` (or pass ``use_fakes=True``) to force in-memory fakes.
When production env vars are set, wiring failures propagate instead of
silently falling back to fakes.
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
    Element,
    IngestPort,
    LegalRegime,
    RetrievalPort,
    VerifierPort,
)

log = logging.getLogger(__name__)

_DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
_DEFAULT_SPARSE_MODEL = "Qdrant/bm25"
_DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


def _production_env_configured() -> bool:
    """True when core Postgres + Qdrant env is set — no silent fake fallback."""
    return bool(os.getenv("DATABASE_URL_SYNC")) and bool(os.getenv("QDRANT_HOST"))


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
    scorer: object | None = None


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
        if _production_env_configured():
            log.error("real bundle wiring failed with production env set — not falling back")
            raise
        log.warning("real bundle wiring failed (%s) — falling back to fakes", e)
        return _build_fake_bundle(config, samples_dir)


def _build_real_bundle(config: ConfigRepositoryPort, samples_dir: Path) -> AdapterBundle:
    """Concrete production adapters wired from env config."""
    from rie_classify import ClassificationService, OllamaClient
    from rie_coverage import CoverageReasoner, Layer2ScoringService
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

    from rie_orchestration.regime import assemble_regime_for_clause

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
        dense_model=os.getenv("EMBEDDING_MODEL", _DEFAULT_EMBEDDING_MODEL),
        sparse_model=os.getenv("SPARSE_MODEL", _DEFAULT_SPARSE_MODEL),
    )
    reranker = BgeReranker(
        model_name=os.getenv("RERANKER_MODEL", _DEFAULT_RERANKER_MODEL),
    )
    retrieval = RetrievalService(
        vector_store=vector_store,
        reranker=reranker,
        dense_embedder=embedder,
        sparse_embedder=embedder,
        collection=os.getenv("QDRANT_COLLECTION", "rie_clauses"),
    )

    llm = OllamaClient(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        model=os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_K_M"),
    )

    def _regime_assembler(clause: Element) -> LegalRegime:
        edges = repo.get_structure_edges(clause.doc_id)
        element_ids = {clause.element_id}
        for edge in edges:
            element_ids.add(edge.from_element)
            element_ids.add(edge.to_element)
        elements_by_id = {
            element.element_id: element for element in repo.get_elements(list(element_ids))
        }
        return assemble_regime_for_clause(clause, elements_by_id, edges)

    classifier = ClassificationService(
        llm=llm,
        get_element=repo.get_element,
        regime_assembler=_regime_assembler,
    )

    verifier = VerificationService(
        nli_model=TransformersNliBackend(
            model_name=os.getenv("NLI_MODEL", "cross-encoder/nli-deberta-v3-base")
        ),
        second_llm=OllamaSecondLlm(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            model=os.getenv("OLLAMA_VERIFIER_MODEL", "mistral:7b-instruct-q4_K_M"),
        ),
        kg_gate_enabled=os.getenv("RIE_KG_GATE_ENABLED") == "1",
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
        scorer=Layer2ScoringService(),
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


