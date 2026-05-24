"""End-to-end `RetrievalService` pipeline test using fakes.

The pipeline under test:
  index_document(meta, elements, edges)
    → chunker → dense + sparse embed → vector-store upsert
  retrieve(query, jurisdiction)
    → embed query → hybrid search → jurisdiction filter
    → cross-encoder rerank → parent + neighbourhood expansion → RetrievalHit
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
    RetrievalPort,
    StructureEdge,
    StructureEdgeType,
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

# ── fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def now() -> datetime:
    return datetime(2026, 5, 24, tzinfo=UTC)


@pytest.fixture
def doc_meta(now: datetime) -> DocumentMeta:
    return DocumentMeta(
        doc_id="sample_dpa_2020",
        jurisdiction="SAMPLE",
        title="Sample Personal Data Protection Act, 2020",
        document_type=DocumentType.STATUTE,
        authority_tier=AuthorityTier.TIER_1_STATUTE,
        sha256="a" * 64,
        retrieved_at=now,
        language="en",
    )


def _make_element(
    eid: str,
    *,
    text: str,
    element_type: ElementType,
    parent_id: str | None,
    char_start: int,
    page: int = 1,
) -> Element:
    return Element(
        element_id=eid,
        doc_id="sample_dpa_2020",
        parent_id=parent_id,
        element_type=element_type,
        text=text,
        page=page,
        char_start=char_start,
        char_end=char_start + len(text),
        extraction_confidence=0.99,
    )


@pytest.fixture
def elements() -> list[Element]:
    heading = _make_element(
        "s26_heading",
        text="Section 26 — Cross-border transfers",
        element_type=ElementType.HEADING,
        parent_id=None,
        char_start=0,
    )
    cross_border = _make_element(
        "s26_p1",
        text=(
            "An organisation shall not transfer any personal data outside the country "
            "unless it has taken appropriate steps to ensure that the recipient is "
            "bound by legally enforceable obligations to provide to the transferred "
            "personal data a standard of protection that is comparable to the "
            "protection under this Act."
        ),
        element_type=ElementType.PARAGRAPH,
        parent_id="s26_heading",
        char_start=200,
    )
    definition_pd = _make_element(
        "s2_def_personal_data",
        text=(
            "Personal data means data, whether true or not, about an individual who can "
            "be identified from that data or from that data and other information."
        ),
        element_type=ElementType.DEFINITION,
        parent_id=None,
        char_start=600,
    )
    consent = _make_element(
        "s13_p1",
        text=(
            "An organisation shall not collect, use or disclose personal data about an "
            "individual unless the individual gives, or is deemed to have given, consent."
        ),
        element_type=ElementType.PARAGRAPH,
        parent_id=None,
        char_start=900,
    )
    breach = _make_element(
        "s26a_p1",
        text=(
            "A data breach must be notified to the Commission as soon as practicable "
            "and in any case not later than seventy-two hours after becoming aware."
        ),
        element_type=ElementType.PARAGRAPH,
        parent_id=None,
        char_start=1200,
    )
    list_item = _make_element(
        "s26_li_1",
        text="binding corporate rules approved by the Commission;",
        element_type=ElementType.LIST_ITEM,
        parent_id="s26_p1",
        char_start=1500,
    )
    return [heading, cross_border, definition_pd, consent, breach, list_item]


@pytest.fixture
def edges() -> list[StructureEdge]:
    return [
        StructureEdge(
            from_element="s26_p1",
            to_element="s2_def_personal_data",
            edge_type=StructureEdgeType.CROSS_REFERENCE,
            raw_reference="personal data",
        ),
        StructureEdge(
            from_element="s26_p1",
            to_element="s26_li_1",
            edge_type=StructureEdgeType.PROVISO,
        ),
    ]


def _build_service() -> RetrievalService:
    return RetrievalService(
        vector_store=InMemoryVectorStore(),
        reranker=IdentityReranker(),
        dense_embedder=DummyDenseEmbedder(),
        sparse_embedder=DummySparseEmbedder(),
        collection="rie_test",
    )


# ── tests ───────────────────────────────────────────────────────────────────


def test_default_top_k_values_match_section_6_1() -> None:
    # §6.1 of systemArchitecture.md: top 20–40 → keep top 6–10.
    assert DEFAULT_TOP_K_INITIAL == 40
    assert DEFAULT_TOP_K_RERANK == 10


def test_service_implements_retrieval_port() -> None:
    assert isinstance(_build_service(), RetrievalPort)


def test_index_then_retrieve_returns_cross_border_clause_first(
    doc_meta: DocumentMeta, elements: list[Element], edges: list[StructureEdge]
) -> None:
    service = _build_service()
    service.index_document(doc_meta, elements, edges)

    hits = service.retrieve(
        query="outbound transfer comparable protection",
        jurisdiction="SAMPLE",
        top_k=3,
    )
    assert hits, "expected at least one hit"
    top = hits[0]
    assert top.element_id == "s26_p1"
    assert top.doc_id == "sample_dpa_2020"
    assert top.parent_element_id == "s26_heading"
    # Neighbourhood expansion: the cross_reference + proviso edges must surface.
    assert "s2_def_personal_data" in top.neighbourhood_element_ids
    assert "s26_li_1" in top.neighbourhood_element_ids
    assert top.snippet
    assert top.rerank_score is not None
    assert top.score == top.rerank_score


def test_headings_are_never_indexed(
    doc_meta: DocumentMeta, elements: list[Element], edges: list[StructureEdge]
) -> None:
    service = _build_service()
    service.index_document(doc_meta, elements, edges)
    hits = service.retrieve(
        query="cross-border transfers section heading", jurisdiction=None, top_k=10
    )
    returned_ids = {h.element_id for h in hits}
    assert "s26_heading" not in returned_ids


def test_jurisdiction_filter_excludes_other_jurisdictions(
    doc_meta: DocumentMeta, elements: list[Element], edges: list[StructureEdge]
) -> None:
    service = _build_service()
    service.index_document(doc_meta, elements, edges)

    other_meta = DocumentMeta(
        doc_id="other_2021",
        jurisdiction="OTHER",
        title="Other Jurisdiction Act",
        document_type=DocumentType.STATUTE,
        authority_tier=AuthorityTier.TIER_1_STATUTE,
        sha256="b" * 64,
        retrieved_at=doc_meta.retrieved_at,
        language="en",
    )
    other_el = _make_element(
        "other_p1",
        text=(
            "An organisation in another jurisdiction shall not transfer personal data "
            "outside the country without comparable protection in place."
        ),
        element_type=ElementType.PARAGRAPH,
        parent_id=None,
        char_start=0,
    )
    # Patch doc_id on the element so the contract is satisfied for the other doc.
    other_el = Element.model_validate(other_el.model_dump() | {"doc_id": "other_2021"})
    service.index_document(other_meta, [other_el], [])

    sample_only = service.retrieve(
        query="outbound transfer comparable protection",
        jurisdiction="SAMPLE",
        top_k=5,
    )
    for hit in sample_only:
        assert hit.doc_id == "sample_dpa_2020"

    no_filter = service.retrieve(
        query="outbound transfer comparable protection",
        jurisdiction=None,
        top_k=5,
    )
    doc_ids = {h.doc_id for h in no_filter}
    # When no filter is applied, both jurisdictions are eligible.
    assert "sample_dpa_2020" in doc_ids
    assert "other_2021" in doc_ids


def test_top_k_caps_returned_hits(
    doc_meta: DocumentMeta, elements: list[Element], edges: list[StructureEdge]
) -> None:
    service = _build_service()
    service.index_document(doc_meta, elements, edges)
    hits = service.retrieve(query="data", jurisdiction=None, top_k=2)
    assert len(hits) <= 2


def test_empty_query_returns_no_hits(
    doc_meta: DocumentMeta, elements: list[Element], edges: list[StructureEdge]
) -> None:
    service = _build_service()
    service.index_document(doc_meta, elements, edges)
    assert list(service.retrieve(query="   ", jurisdiction=None, top_k=5)) == []


def test_retrieval_hit_carries_dense_and_rerank_scores(
    doc_meta: DocumentMeta, elements: list[Element], edges: list[StructureEdge]
) -> None:
    service = _build_service()
    service.index_document(doc_meta, elements, edges)
    hits = service.retrieve(
        query="outbound transfer comparable protection", jurisdiction="SAMPLE", top_k=3
    )
    for hit in hits:
        assert hit.rerank_score is not None
        # Dense score may legitimately be None if the child wasn't in the raw
        # hit list (e.g. only-sparse match), but with the dummy embedder both
        # legs see every doc.
        assert hit.dense_score is not None
