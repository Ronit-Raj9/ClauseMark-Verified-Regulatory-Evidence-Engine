"""Contract tests for `RetrievalPort` (+ `VectorStorePort`, `RerankerPort`).

Adapter-agnostic — anything passing this test is "shaped right". The fakes
shipped in `rie_retrieval.fakes` are the runtime stand-ins used here; the
production Qdrant + BGE-M3 + cross-encoder stack must satisfy the same shape.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from rie_contracts import (
    AuthorityTier,
    DocumentMeta,
    DocumentType,
    Element,
    ElementType,
    RerankerPort,
    RetrievalHit,
    RetrievalPort,
    StructureEdge,
    StructureEdgeType,
    VectorStorePort,
)
from rie_retrieval import (
    DEFAULT_TOP_K_INITIAL,
    DEFAULT_TOP_K_RERANK,
    DummyDenseEmbedder,
    DummySparseEmbedder,
    IdentityReranker,
    InMemoryVectorStore,
    RetrievalService,
)

# ── runtime isinstance checks (the cheap "shape" gate) ─────────────────────


def test_in_memory_vector_store_implements_port() -> None:
    assert isinstance(InMemoryVectorStore(), VectorStorePort)


def test_identity_reranker_implements_port() -> None:
    assert isinstance(IdentityReranker(), RerankerPort)


def test_retrieval_service_implements_port() -> None:
    service = RetrievalService(
        vector_store=InMemoryVectorStore(),
        reranker=IdentityReranker(),
        dense_embedder=DummyDenseEmbedder(),
        sparse_embedder=DummySparseEmbedder(),
        collection="contract_test",
    )
    assert isinstance(service, RetrievalPort)


def test_defaults_track_section_6_1_top_k_window() -> None:
    # §6.1: retrieve top 20–40, rerank down to top 6–10. We pick the upper end.
    assert 20 <= DEFAULT_TOP_K_INITIAL <= 40
    assert 6 <= DEFAULT_TOP_K_RERANK <= 10
    assert DEFAULT_TOP_K_RERANK <= DEFAULT_TOP_K_INITIAL


# ── tiny E2E using the shipped fakes ───────────────────────────────────────


@pytest.fixture
def doc_meta() -> DocumentMeta:
    return DocumentMeta(
        doc_id="contract_dpa",
        jurisdiction="SAMPLE",
        title="Contract-test DPA",
        document_type=DocumentType.STATUTE,
        authority_tier=AuthorityTier.TIER_1_STATUTE,
        sha256="c" * 64,
        retrieved_at=datetime(2026, 5, 24, tzinfo=UTC),
        language="en",
    )


def _el(eid: str, *, text: str, et: ElementType, parent: str | None, start: int) -> Element:
    return Element(
        element_id=eid,
        doc_id="contract_dpa",
        parent_id=parent,
        element_type=et,
        text=text,
        page=1,
        char_start=start,
        char_end=start + len(text),
        extraction_confidence=0.95,
    )


@pytest.fixture
def elements() -> list[Element]:
    return [
        _el(
            "heading_xb",
            text="Cross-border transfers",
            et=ElementType.HEADING,
            parent=None,
            start=0,
        ),
        _el(
            "p_xb",
            text=(
                "An organisation shall not transfer personal data outside the country "
                "unless the recipient provides comparable protection."
            ),
            et=ElementType.PARAGRAPH,
            parent="heading_xb",
            start=100,
        ),
        _el(
            "p_consent",
            text=(
                "An organisation shall not collect personal data unless the individual "
                "has given valid consent."
            ),
            et=ElementType.PARAGRAPH,
            parent=None,
            start=400,
        ),
        _el(
            "def_pd",
            text="Personal data means information about an identified individual.",
            et=ElementType.DEFINITION,
            parent=None,
            start=700,
        ),
    ]


@pytest.fixture
def edges() -> list[StructureEdge]:
    return [
        StructureEdge(
            from_element="p_xb",
            to_element="def_pd",
            edge_type=StructureEdgeType.CROSS_REFERENCE,
        ),
    ]


def test_retrieval_e2e_with_fakes(
    doc_meta: DocumentMeta, elements: list[Element], edges: list[StructureEdge]
) -> None:
    service = RetrievalService(
        vector_store=InMemoryVectorStore(),
        reranker=IdentityReranker(),
        dense_embedder=DummyDenseEmbedder(),
        sparse_embedder=DummySparseEmbedder(),
        collection="contract_e2e",
    )
    service.index_document(doc_meta, elements, edges)

    hits = service.retrieve(
        query="outbound transfer comparable protection",
        jurisdiction="SAMPLE",
        top_k=3,
    )
    assert hits, "expected at least one hit from fakes pipeline"
    assert all(isinstance(h, RetrievalHit) for h in hits)
    top = hits[0]
    assert top.element_id == "p_xb"
    assert top.doc_id == "contract_dpa"
    assert top.parent_element_id == "heading_xb"
    assert "def_pd" in top.neighbourhood_element_ids
    assert top.snippet
    assert top.rerank_score is not None


def test_retrieve_backcompat_without_new_kwargs(
    doc_meta: DocumentMeta, elements: list[Element], edges: list[StructureEdge]
) -> None:
    # Phase 2 added optional `query_expander` ctor arg + `keywords_by_lang`
    # retrieve kwarg. Existing callers that pass NEITHER must see byte-identical
    # behaviour: the positional `retrieve(query, jurisdiction, top_k)` API and
    # its results are unchanged.
    service = RetrievalService(
        vector_store=InMemoryVectorStore(),
        reranker=IdentityReranker(),
        dense_embedder=DummyDenseEmbedder(),
        sparse_embedder=DummySparseEmbedder(),
        collection="contract_backcompat",
    )
    service.index_document(doc_meta, elements, edges)

    hits = service.retrieve("outbound transfer comparable protection", "SAMPLE", 3)
    assert hits
    assert hits[0].element_id == "p_xb"
    # Passing keywords_by_lang with NO expander wired is a safe no-op (identical).
    hits_kw = service.retrieve(
        "outbound transfer comparable protection",
        "SAMPLE",
        3,
        keywords_by_lang={"fr": ["transfert transfrontalier"]},
    )
    assert [h.element_id for h in hits_kw] == [h.element_id for h in hits]
