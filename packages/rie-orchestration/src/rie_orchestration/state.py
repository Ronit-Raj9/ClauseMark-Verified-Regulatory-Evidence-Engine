"""LangGraph state model for the RIE pipeline.

Each node receives + mutates a typed `RieState`. State persists to the
LangGraph checkpointer (Postgres in production, in-memory in tests).
"""

from __future__ import annotations

from typing import TypedDict

from rie_contracts import (
    Claim,
    CoverageRecord,
    DocumentMeta,
    Element,
    Layer2Recommendation,
    RetrievalHit,
    StructureEdge,
    VerificationReport,
)


class RieState(TypedDict, total=False):
    # Run scope
    run_id: str
    jurisdiction: str
    pillar_ids: list[str]
    dry_run: bool
    skip_hitl: bool
    local_only: bool

    # Ingest output
    documents: list[DocumentMeta]
    raw_bytes_by_doc: dict[str, bytes]

    # Extract output
    elements_by_doc: dict[str, list[Element]]
    edges_by_doc: dict[str, list[StructureEdge]]

    # Retrieval output
    candidate_hits_by_pillar: dict[str, list[RetrievalHit]]

    # Classification + verification output
    claims: list[Claim]
    verifications: dict[str, VerificationReport]  # claim_id → report

    # Coverage output
    coverage: list[CoverageRecord]

    # Layer-2 recommendations (never final scores)
    layer2_recommendations: list[Layer2Recommendation]

    # Graceful degradation (§11)
    ingest_degraded: bool

    # Aggregates
    errors: list[str]
    notes: list[str]
