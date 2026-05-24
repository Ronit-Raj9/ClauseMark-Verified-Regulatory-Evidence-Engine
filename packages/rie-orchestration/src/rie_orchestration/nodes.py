"""Pipeline nodes — pure functions over (state, bundle) → state.

LangGraph wires these into a directed graph; each node is idempotent given the
same input state so the checkpointer can resume safely.
"""

from __future__ import annotations

import logging
import os

from rie_contracts import (
    Claim,
    CoverageRecord,
    DocumentMeta,
    DocumentType,
    Element,
    RetrievalHit,
    StructureEdge,
    VerificationReport,
    VerificationStatus,
)

from rie_orchestration.state import RieState
from rie_orchestration.wiring import AdapterBundle

log = logging.getLogger(__name__)


def ingest_node(state: RieState, bundle: AdapterBundle) -> RieState:
    jurisdiction = state["jurisdiction"]
    log.info("ingest: jurisdiction=%s", jurisdiction)
    entries = bundle.ingest.load_source_registry(jurisdiction)
    documents: list[DocumentMeta] = []
    raw_by_doc: dict[str, bytes] = {}
    for entry in entries:
        meta, raw = bundle.ingest.load_document_bytes(entry)
        documents.append(meta)
        raw_by_doc[meta.doc_id] = raw
        bundle.repo.save_document(meta)
    state["documents"] = documents
    state["raw_bytes_by_doc"] = raw_by_doc
    return state


def extract_node(state: RieState, bundle: AdapterBundle) -> RieState:
    elements_by_doc: dict[str, list[Element]] = {}
    edges_by_doc: dict[str, list[StructureEdge]] = {}
    for meta in state.get("documents", []):
        raw = state["raw_bytes_by_doc"][meta.doc_id]
        elements, edges = bundle.extractor.extract(meta, raw)
        elements_by_doc[meta.doc_id] = list(elements)
        edges_by_doc[meta.doc_id] = list(edges)
        bundle.repo.save_elements(meta.doc_id, elements, edges)
        log.info("extract: doc=%s elements=%d edges=%d", meta.doc_id, len(elements), len(edges))
    state["elements_by_doc"] = elements_by_doc
    state["edges_by_doc"] = edges_by_doc
    return state


def index_node(state: RieState, bundle: AdapterBundle) -> RieState:
    for meta in state.get("documents", []):
        elements = state["elements_by_doc"][meta.doc_id]
        edges = state["edges_by_doc"][meta.doc_id]
        bundle.retrieval.index_document(meta, elements, edges)
    return state


def retrieve_node(state: RieState, bundle: AdapterBundle) -> RieState:
    """For each (pillar, indicator), build a query and retrieve candidates."""

    hits_by_pillar: dict[str, list[RetrievalHit]] = {}
    pillars = bundle.config.load_registry()
    selected = {p for p in state["pillar_ids"]}
    for entry in pillars:
        if entry.pillar_id not in selected:
            continue
        if entry.status != "built":
            log.info("skip pillar %s (status=%s)", entry.pillar_id, entry.status)
            continue
        pillar = bundle.config.load_pillar(entry.pillar_id)
        pillar_hits: list[RetrievalHit] = []
        for ind in pillar.indicators:
            keywords = []
            for lang_kws in ind.positive_keywords.values():
                keywords.extend(lang_kws)
            query = f"{ind.name}. {ind.definition}. {' '.join(keywords)}"
            hits = bundle.retrieval.retrieve(
                query=query, jurisdiction=state["jurisdiction"], top_k=8
            )
            pillar_hits.extend(hits)
        hits_by_pillar[entry.pillar_id] = pillar_hits
        log.info("retrieve: pillar=%s hits=%d", entry.pillar_id, len(pillar_hits))
    state["candidate_hits_by_pillar"] = hits_by_pillar
    return state


def classify_node(state: RieState, bundle: AdapterBundle) -> RieState:
    """Run the LLM classifier on each candidate clause."""

    claims: list[Claim] = []
    pillars_loaded = {p: bundle.config.load_pillar(p) for p in state["pillar_ids"]}
    seen: set[str] = set()
    for pillar_id, hits in state.get("candidate_hits_by_pillar", {}).items():
        pillar = pillars_loaded[pillar_id]
        for hit in hits:
            if hit.parent_element_id in seen:
                continue
            seen.add(hit.parent_element_id)
            try:
                clause = bundle.repo.get_element(hit.parent_element_id)
            except KeyError:
                log.warning("classify: missing element %s", hit.parent_element_id)
                continue
            neighbourhood = bundle.repo.get_elements(list(hit.neighbourhood_element_ids))
            try:
                n_samples = int(os.getenv("RIE_N_SAMPLES", "3"))
                claim = bundle.classifier.classify_clause(
                    clause_element=clause,
                    neighbourhood=neighbourhood,
                    pillar=pillar,
                    indicator_choices=pillar.indicators,
                    n_samples=n_samples,
                )
            except Exception as e:
                log.warning(
                    "classify: skipping element %s — %s: %s",
                    hit.parent_element_id,
                    type(e).__name__,
                    e,
                )
                state.setdefault("errors", []).append(
                    f"classify {hit.parent_element_id}: {type(e).__name__}"
                )
                continue
            if claim.jurisdiction != state["jurisdiction"]:
                claim = claim.model_copy(update={"jurisdiction": state["jurisdiction"]})
            bundle.repo.save_claim(claim)
            claims.append(claim)
    state["claims"] = claims
    log.info("classify: produced %d claims", len(claims))
    return state


def verify_node(state: RieState, bundle: AdapterBundle) -> RieState:
    verifications: dict[str, VerificationReport] = {}
    kg_enabled = os.getenv("RIE_KG_GATE_ENABLED") == "1"
    kg_results: dict[str, object] = {}
    for claim in state.get("claims", []):
        if kg_enabled and hasattr(bundle.verifier, "verify_with_kg"):
            report, kg_result = bundle.verifier.verify_with_kg(  # type: ignore[attr-defined]
                claim, bundle.repo.get_element_text
            )
            if kg_result is not None:
                kg_results[claim.claim_id] = kg_result
        else:
            report = bundle.verifier.verify(claim, bundle.repo.get_element_text)
        bundle.repo.save_verification(report)
        verifications[claim.claim_id] = report
    state["verifications"] = verifications
    if kg_results:
        state["kg_results"] = kg_results
    log.info(
        "verify: %d verified / %d flagged / %d rejected%s",
        sum(1 for r in verifications.values() if r.status == VerificationStatus.VERIFIED),
        sum(1 for r in verifications.values() if r.status == VerificationStatus.FLAGGED),
        sum(1 for r in verifications.values() if r.status == VerificationStatus.REJECTED),
        f" (kg_gate={len(kg_results)} side-channel)" if kg_results else "",
    )
    return state


def coverage_node(state: RieState, bundle: AdapterBundle) -> RieState:
    coverage: list[CoverageRecord] = []
    enriched_rows: list[object] = []
    pillars_loaded = {p: bundle.config.load_pillar(p) for p in state["pillar_ids"]}
    verified = [
        c
        for c in state.get("claims", [])
        if state["verifications"].get(c.claim_id)
        and state["verifications"][c.claim_id].status == VerificationStatus.VERIFIED
    ]

    corpus_completeness_enabled = os.getenv("RIE_CORPUS_COMPLETENESS_ENABLED") == "1"
    manifest = None
    ingested_types: list[DocumentType] = []
    if corpus_completeness_enabled:
        from rie_coverage import enrich_with_completeness, load_manifest

        repo_root = bundle.samples_dir.parent.parent
        manifest = load_manifest(repo_root, state["jurisdiction"])
        seen_types: set[DocumentType] = set()
        for doc in state.get("documents", []):
            if doc.document_type not in seen_types:
                seen_types.add(doc.document_type)
                ingested_types.append(doc.document_type)

    for pillar_id, pillar in pillars_loaded.items():
        for ind in pillar.indicators:
            rec = bundle.coverage.evaluate(
                jurisdiction=state["jurisdiction"],
                indicator_id=ind.indicator_id,
                verified_claims=verified,
                gold_recall=_lookup_gold_recall(bundle, pillar_id, ind.indicator_id),
            )
            if corpus_completeness_enabled and manifest is not None:
                enriched = enrich_with_completeness(
                    rec,
                    ingested_source_types=ingested_types,
                    manifest=manifest,
                )
                enriched_rows.append(enriched)
                rec = enriched.to_contract_row()
            bundle.repo.save_coverage(rec)
            coverage.append(rec)
    state["coverage"] = coverage
    if enriched_rows:
        state["enriched_coverage"] = enriched_rows
    log.info(
        "coverage: produced %d records%s",
        len(coverage),
        " (corpus completeness)" if enriched_rows else "",
    )
    return state


def _lookup_gold_recall(bundle: AdapterBundle, pillar_id: str, indicator_id: str) -> float | None:
    try:
        gold = bundle.config.load_gold(pillar_id)
    except Exception:
        return None
    matching = [g for g in gold if g.indicator_id == indicator_id]
    if not matching:
        return None
    return 0.82  # placeholder; real value comes from eval pipeline
