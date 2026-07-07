"""`RetrievalService` — parent-document retrieval with hybrid + rerank.

Implements `RetrievalPort` (§6.1 of `systemArchitecture.md`):

  1. Index time — split leaf elements (paragraph / list_item / definition; never
     headings) into ~480-character child chunks, embed each with a dense
     (BGE-M3) and a sparse (BM25) vector, and upsert into the configured
     `VectorStorePort`. Each child's payload carries everything the retrieval
     side needs to reconstruct the parent element and its structure-graph
     neighbourhood — no second datastore lookup.
  2. Query time — embed the query, hybrid-search via the store (which fuses
     the two legs with RRF k=60 in Qdrant), filter by jurisdiction, cross-
     encoder rerank the top_k_initial down to top_k_rerank, then expand each
     surviving child into a `RetrievalHit` carrying its parent element id and
     graph neighbourhood.

Multilingual notes (Phase 2):
  * BGE-M3 is already multilingual — the dense leg needs no per-language
    handling beyond passing the hint through for telemetry parity.
  * The sparse leg uses `rie_retrieval.tokenizer.tokenize`, which
    NFKC-normalises, splits on unicode whitespace+punct, handles FR-style
    elision, and falls back to CJK character bigrams when a token has no
    word boundary.
  * Documents are indexed under `doc_meta.language`; queries take an
    optional `lang` arg (added as a keyword-only parameter — the original
    positional signature on `RetrievalPort.retrieve` is untouched so the
    frozen contract still holds).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

from rie_contracts import (
    DocumentMeta,
    Element,
    ElementType,
    RerankerPort,
    RetrievalHit,
    RetrievalPort,
    StructureEdge,
    VectorStorePort,
)

from rie_retrieval.chunker import chunk_text
from rie_retrieval.embedder import (
    BGE_M3_DENSE_DIM,
    DenseEmbedderPort,
    SparseEmbedderPort,
)
from rie_retrieval.multilingual import LanguageAwareQueryExpander, detect_language

__all__ = [
    "DEFAULT_CHILD_CHUNK_OVERLAP",
    "DEFAULT_CHILD_CHUNK_SIZE",
    "DEFAULT_TOP_K_INITIAL",
    "DEFAULT_TOP_K_RERANK",
    "RetrievalService",
]


# Element types that carry actual legal substance. Headings (and their cousins
# article / section / table containers) are addressable but their text is
# usually a title — embedding them pollutes recall, so they're skipped at index
# time. This matches the "child chunks for matching" half of small-to-big.
_LEAF_ELEMENT_TYPES: frozenset[ElementType] = frozenset(
    {ElementType.PARAGRAPH, ElementType.LIST_ITEM, ElementType.DEFINITION}
)

# Defaults per task brief (§6.1: top 20–40 → keep 6–10; chunk ≈ 480 chars).
DEFAULT_TOP_K_INITIAL: int = 40
DEFAULT_TOP_K_RERANK: int = 10
DEFAULT_CHILD_CHUNK_SIZE: int = 480
DEFAULT_CHILD_CHUNK_OVERLAP: int = 64
_SNIPPET_PREVIEW_MAX = 320


@dataclass(frozen=True)
class _IndexedDocument:
    """In-memory bookkeeping for one indexed document.

    Kept alongside the vector store so retrieval can (a) reconstruct the
    parent element from a child id without a DB call, (b) hand back the
    structure-graph neighbourhood the LLM needs for whole-law reasoning, and
    (c) route queries to the same tokenizer settings the document was
    indexed under (`language`).
    """

    doc_id: str
    jurisdiction: str
    language: str
    parent_of: dict[str, str]  # element_id -> parent_id (or self)
    neighbours_of: dict[str, list[str]]  # element_id -> sorted unique neighbours


@dataclass(frozen=True)
class _RetrievalQuery:
    """Internal query carrier — extends the port's positional API with `lang`.

    Lives here (NOT in `rie-contracts`) deliberately: the frozen contract
    can't grow new fields, but the service is free to enrich what it actually
    operates on. Constructed at the top of `retrieve(...)` so the rest of the
    pipeline only touches this typed object.
    """

    text: str
    jurisdiction: str | None
    lang: str | None
    top_k: int


class RetrievalService(RetrievalPort):
    """Parent-document retrieval with hybrid + cross-encoder rerank."""

    def __init__(
        self,
        vector_store: VectorStorePort,
        reranker: RerankerPort,
        dense_embedder: DenseEmbedderPort,
        sparse_embedder: SparseEmbedderPort,
        collection: str,
        *,
        top_k_initial: int = DEFAULT_TOP_K_INITIAL,
        top_k_rerank: int = DEFAULT_TOP_K_RERANK,
        child_chunk_size: int = DEFAULT_CHILD_CHUNK_SIZE,
        child_chunk_overlap: int = DEFAULT_CHILD_CHUNK_OVERLAP,
        dense_dim: int = BGE_M3_DENSE_DIM,
        query_expander: LanguageAwareQueryExpander | None = None,
    ) -> None:
        if top_k_rerank > top_k_initial:
            msg = f"top_k_rerank ({top_k_rerank}) must be <= top_k_initial ({top_k_initial})"
            raise ValueError(msg)
        self._store = vector_store
        self._reranker = reranker
        self._dense = dense_embedder
        self._sparse = sparse_embedder
        self._collection = collection
        self._top_k_initial = top_k_initial
        self._top_k_rerank = top_k_rerank
        self._chunk_size = child_chunk_size
        self._chunk_overlap = child_chunk_overlap
        self._dense_dim = dense_dim
        self._query_expander = query_expander
        self._docs: dict[str, _IndexedDocument] = {}
        # element_id -> the rendered text we hand to the reranker as the
        # candidate document, plus its char offsets for snippet reconstruction.
        self._child_text: dict[str, tuple[str, int, int]] = {}
        self._collection_ready = False

    # ── RetrievalPort: index_document ────────────────────────────────────

    def index_document(
        self,
        doc_meta: DocumentMeta,
        elements: Sequence[Element],
        edges: Sequence[StructureEdge],
    ) -> None:
        if not self._collection_ready:
            self._store.ensure_collection(self._collection, self._dense_dim)
            self._collection_ready = True

        parent_of, neighbours_of = self._build_parent_and_neighbour_maps(elements, edges)

        leaves = [e for e in elements if e.element_type in _LEAF_ELEMENT_TYPES]

        child_ids: list[str] = []
        child_texts: list[str] = []
        child_payloads: list[dict[str, str | int | float | bool | None]] = []

        # Prefer the declared language; otherwise detect it best-effort from the
        # leaf text so a multilingual collection still tags each child with the
        # right script/lang (used for sparse tokenization + cross-lingual
        # telemetry). Detection never raises and defaults to "en".
        doc_language = doc_meta.language
        if not doc_language:
            sample = " ".join(e.text for e in leaves[:8])
            detected = detect_language(sample)
            doc_language = detected if detected != "unknown" else "en"

        for el in leaves:
            parent_element_id = parent_of.get(el.element_id, el.element_id)
            neighbours = neighbours_of.get(el.element_id, [])
            chunks = chunk_text(el.text, self._chunk_size, self._chunk_overlap)
            if not chunks:
                continue
            for idx, (text, c_start, c_end) in enumerate(chunks):
                child_id = f"{el.element_id}__c{idx}"
                abs_start = el.char_start + c_start
                abs_end = el.char_start + c_end
                snippet = text[:_SNIPPET_PREVIEW_MAX]
                payload: dict[str, str | int | float | bool | None] = {
                    "doc_id": el.doc_id,
                    "element_id": el.element_id,
                    "parent_element_id": parent_element_id,
                    "jurisdiction": doc_meta.jurisdiction,
                    "lang": doc_language,
                    "char_start": abs_start,
                    "char_end": abs_end,
                    "snippet": snippet,
                    # Stored as JSON so the payload value type stays a scalar
                    # primitive (per VectorStorePort signature) while still
                    # carrying the variable-length neighbourhood list.
                    "neighbour_ids_json": json.dumps(neighbours),
                }
                child_ids.append(child_id)
                child_texts.append(text)
                child_payloads.append(payload)
                self._child_text[child_id] = (text, abs_start, abs_end)

        if not child_ids:
            self._docs[doc_meta.doc_id] = _IndexedDocument(
                doc_id=doc_meta.doc_id,
                jurisdiction=doc_meta.jurisdiction,
                language=doc_language,
                parent_of=parent_of,
                neighbours_of=neighbours_of,
            )
            return

        dense_vecs = self._dense.embed_dense(child_texts, lang=doc_language)
        sparse_vecs = self._sparse.embed_sparse(child_texts, lang=doc_language)

        self._store.upsert(
            collection=self._collection,
            ids=child_ids,
            dense_vectors=dense_vecs,
            sparse_vectors=sparse_vecs,
            payloads=child_payloads,
        )

        self._docs[doc_meta.doc_id] = _IndexedDocument(
            doc_id=doc_meta.doc_id,
            jurisdiction=doc_meta.jurisdiction,
            language=doc_language,
            parent_of=parent_of,
            neighbours_of=neighbours_of,
        )

    # ── RetrievalPort: retrieve ──────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        jurisdiction: str | None,
        top_k: int = 8,
        *,
        lang: str | None = None,
        keywords_by_lang: dict[str, list[str]] | None = None,
    ) -> list[RetrievalHit]:
        if not query.strip():
            return []

        # Cross-lingual query expansion (Phase 2): when an expander is wired AND
        # the caller supplies a pillar indicator's multilingual keyword variants,
        # append the other-language surface forms so a query in language A can
        # surface clauses indexed in language B. Default behaviour (no expander
        # OR no keywords) is byte-for-byte unchanged — the original `query` is
        # embedded as-is. Reranking always uses the ORIGINAL `query` text.
        search_text = query
        if self._query_expander is not None and keywords_by_lang:
            search_text = self._query_expander.expand(query, keywords_by_lang)

        q = _RetrievalQuery(text=search_text, jurisdiction=jurisdiction, lang=lang, top_k=top_k)

        dense_q = self._dense.embed_dense([q.text], lang=q.lang)[0]
        sparse_q = self._sparse.embed_sparse([q.text], lang=q.lang)[0]

        # Over-fetch so jurisdiction filtering still leaves enough material to
        # rerank. 4x is the common heuristic — cheap, and the rerank cost is
        # capped at top_k_initial anyway.
        raw_hits = self._store.search_hybrid(
            collection=self._collection,
            dense_query=dense_q,
            sparse_query=sparse_q,
            top_k=max(self._top_k_initial * 4, self._top_k_initial),
        )

        dense_score_by_id = {cid: score for cid, score in raw_hits}

        # Jurisdiction + still-known filter; preserve store-returned ranking.
        filtered: list[tuple[str, float]] = []
        for cid, score in raw_hits:
            if cid not in self._child_text:
                continue
            if jurisdiction is not None:
                payload_jur = self._jurisdiction_for(cid)
                if payload_jur is not None and payload_jur != jurisdiction:
                    continue
            filtered.append((cid, score))
            if len(filtered) >= self._top_k_initial:
                break

        if not filtered:
            return []

        rerank_candidates: list[tuple[str, str]] = [
            (cid, self._child_text[cid][0]) for cid, _ in filtered
        ]
        reranked = list(self._reranker.rerank(query, rerank_candidates))
        keep = reranked[: max(self._top_k_rerank, top_k)]

        hits: list[RetrievalHit] = []
        for cid, rerank_score in keep[:top_k]:
            hit = self._build_hit(
                child_id=cid,
                dense_score=dense_score_by_id.get(cid),
                rerank_score=float(rerank_score),
            )
            if hit is not None:
                hits.append(hit)
        return hits

    # ── helpers ──────────────────────────────────────────────────────────

    def _build_parent_and_neighbour_maps(
        self,
        elements: Sequence[Element],
        edges: Sequence[StructureEdge],
    ) -> tuple[dict[str, str], dict[str, list[str]]]:
        known_ids = {e.element_id for e in elements}
        parent_of: dict[str, str] = {}
        for el in elements:
            parent_of[el.element_id] = (
                el.parent_id if el.parent_id and el.parent_id in known_ids else el.element_id
            )
        neighbour_sets: dict[str, set[str]] = {eid: set() for eid in known_ids}
        for edge in edges:
            if edge.from_element in neighbour_sets and edge.to_element in known_ids:
                if edge.to_element != edge.from_element:
                    neighbour_sets[edge.from_element].add(edge.to_element)
            if edge.to_element in neighbour_sets and edge.from_element in known_ids:
                if edge.from_element != edge.to_element:
                    neighbour_sets[edge.to_element].add(edge.from_element)
        return parent_of, {k: sorted(v) for k, v in neighbour_sets.items()}

    def _jurisdiction_for(self, child_id: str) -> str | None:
        # Resolve the originating element to a jurisdiction via the in-memory
        # doc index. Falls back to None if we don't recognise the id (e.g. an
        # orphaned point from a previous run).
        element_id = child_id.rsplit("__c", 1)[0]
        for doc in self._docs.values():
            if element_id in doc.parent_of:
                return doc.jurisdiction
        return None

    def _build_hit(
        self,
        child_id: str,
        dense_score: float | None,
        rerank_score: float,
    ) -> RetrievalHit | None:
        element_id = child_id.rsplit("__c", 1)[0]
        text, _abs_start, _abs_end = self._child_text[child_id]
        owning_doc = None
        for doc in self._docs.values():
            if element_id in doc.parent_of:
                owning_doc = doc
                break
        if owning_doc is None:
            return None
        parent_id = owning_doc.parent_of.get(element_id, element_id)
        neighbours = list(owning_doc.neighbours_of.get(element_id, []))
        snippet = text[:_SNIPPET_PREVIEW_MAX]
        return RetrievalHit(
            element_id=element_id,
            doc_id=owning_doc.doc_id,
            score=float(rerank_score),
            parent_element_id=parent_id,
            neighbourhood_element_ids=neighbours,
            snippet=snippet,
            dense_score=dense_score,
            sparse_score=None,
            rerank_score=float(rerank_score),
        )
