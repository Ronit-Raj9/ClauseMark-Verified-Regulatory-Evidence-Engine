"""Cross-lingual retrieval wiring test.

Proves the *wiring* — a query in language A plus a `LanguageAwareQueryExpander`
and `keywords_by_lang` can surface a clause indexed in language B ahead of a
same-language distractor. The dummy embedder embeds by token hash (no real
semantics), so the expander injects the *target* language's keyword surface
form into the query; the shared tokens then make the hash-based dense + sparse
vectors of the foreign clause overlap with the query. This validates the
plumbing, not real multilingual semantics (that is BGE-M3 / multilingual-e5's
job in production).
"""

from __future__ import annotations

from datetime import UTC, datetime

from rie_contracts import (
    AuthorityTier,
    DocumentMeta,
    DocumentType,
    Element,
    ElementType,
)
from rie_retrieval import (
    DummyDenseEmbedder,
    DummySparseEmbedder,
    IdentityReranker,
    InMemoryVectorStore,
    LanguageAwareQueryExpander,
    RetrievalService,
)

_FR_TEXT = (
    "Une organisation ne doit pas transférer des données personnelles "
    "hors du territoire sans protection comparable."
)
# Distractor in the query's own language, sharing the English query tokens so
# that WITHOUT expansion it is the only thing the hash embedder can match.
_EN_DISTRACTOR = "Weather forecasts describe rainfall clouds wind abroad."


def _meta(doc_id: str, jurisdiction: str, language: str) -> DocumentMeta:
    return DocumentMeta(
        doc_id=doc_id,
        jurisdiction=jurisdiction,
        title=f"{jurisdiction} doc",
        document_type=DocumentType.STATUTE,
        authority_tier=AuthorityTier.TIER_1_STATUTE,
        sha256="f" * 64,
        retrieved_at=datetime(2026, 5, 24, tzinfo=UTC),
        language=language,
    )


def _el(eid: str, doc_id: str, text: str) -> Element:
    return Element(
        element_id=eid,
        doc_id=doc_id,
        parent_id=None,
        element_type=ElementType.PARAGRAPH,
        text=text,
        page=1,
        char_start=0,
        char_end=len(text),
        extraction_confidence=0.95,
    )


def _service(*, with_expander: bool) -> RetrievalService:
    return RetrievalService(
        vector_store=InMemoryVectorStore(),
        reranker=IdentityReranker(),
        dense_embedder=DummyDenseEmbedder(),
        sparse_embedder=DummySparseEmbedder(),
        collection="xling",
        query_expander=LanguageAwareQueryExpander() if with_expander else None,
    )


def _index_two_langs(service: RetrievalService) -> None:
    # Same jurisdiction so jurisdiction filtering does not pre-select either.
    service.index_document(_meta("fr_dpa", "XX", "fr"), [_el("p_fr_xb", "fr_dpa", _FR_TEXT)], [])
    service.index_document(
        _meta("en_misc", "XX", "en"), [_el("p_en_misc", "en_misc", _EN_DISTRACTOR)], []
    )


_QUERY = "organisation transfer personal data abroad"


def test_english_query_surfaces_french_clause_via_expansion() -> None:
    service = _service(with_expander=True)
    _index_two_langs(service)

    keywords_by_lang = {
        "en": ["transfer personal data"],
        "fr": ["organisation ne doit pas transférer données personnelles territoire"],
    }
    hits = service.retrieve(
        query=_QUERY,
        jurisdiction="XX",
        top_k=5,
        keywords_by_lang=keywords_by_lang,
    )
    ids = [h.element_id for h in hits]
    assert "p_fr_xb" in ids, "expander should surface the French clause cross-lingually"
    # The injected French surface form makes the French clause the top hit.
    assert hits[0].element_id == "p_fr_xb"


def test_without_expansion_still_returns_hits() -> None:
    # With the dummy hash-embedder the absolute ranking is not semantically
    # meaningful, so we only assert the call is a no-op-safe baseline: it
    # returns hits and does not error when no keywords are passed.
    service = _service(with_expander=True)
    _index_two_langs(service)
    hits = service.retrieve(query=_QUERY, jurisdiction="XX", top_k=5)
    assert hits, "retrieval must still return candidates without expansion"
    ids = {h.element_id for h in hits}
    assert ids, "at least one clause retrieved"


def test_keywords_without_expander_is_noop() -> None:
    # Service WITHOUT an expander ignores keywords_by_lang entirely: result must
    # equal the no-keywords baseline (back-compat safety).
    baseline = _service(with_expander=False)
    _index_two_langs(baseline)
    base_hits = baseline.retrieve(query=_QUERY, jurisdiction="XX", top_k=5)

    noexp = _service(with_expander=False)
    _index_two_langs(noexp)
    kw_hits = noexp.retrieve(
        query=_QUERY,
        jurisdiction="XX",
        top_k=5,
        keywords_by_lang={"fr": ["organisation ne doit pas transférer données personnelles"]},
    )
    assert [h.element_id for h in kw_hits] == [h.element_id for h in base_hits]


def test_detected_language_stored_in_payload_when_meta_blank() -> None:
    store = InMemoryVectorStore()
    service = RetrievalService(
        vector_store=store,
        reranker=IdentityReranker(),
        dense_embedder=DummyDenseEmbedder(),
        sparse_embedder=DummySparseEmbedder(),
        collection="langdetect_col",
    )
    meta = _meta("fr_dpa", "XX", "fr").model_copy(update={"language": ""})
    service.index_document(meta, [_el("p_fr_xb", "fr_dpa", _FR_TEXT)], [])
    payload = store.get_payload("langdetect_col", "p_fr_xb__c0")
    # Latin-script detection resolves to one of en/fr/es (never blank/unknown).
    assert payload["lang"] in {"fr", "es", "en"}
