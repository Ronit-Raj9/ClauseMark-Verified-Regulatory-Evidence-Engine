"""High-level runner — the entry point used by the CLI + the API."""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rie_contracts import VerificationStatus
from rie_orchestration.state import RieState
from rie_orchestration.tracing import LangfuseTracer
from rie_orchestration.wiring import AdapterBundle, build_default_bundle

log = logging.getLogger(__name__)


@dataclass
class RunOptions:
    jurisdiction: str
    pillar_ids: list[str]
    dry_run: bool = False
    repo_root: Path | None = None
    bundle: AdapterBundle | None = None
    use_fakes: bool = False
    run_id: str | None = field(default=None)
    enable_hitl: bool = True
    enable_postgres_checkpointer: bool | None = None  # None = auto from env


def run_pipeline(
    jurisdiction: str,
    pillar_ids: list[str],
    *,
    dry_run: bool = False,
    repo_root: Path | None = None,
    bundle: AdapterBundle | None = None,
    use_fakes: bool = False,
    run_id: str | None = None,
    enable_hitl: bool = True,
    enable_postgres_checkpointer: bool | None = None,
) -> dict[str, Any]:
    options = RunOptions(
        jurisdiction=jurisdiction,
        pillar_ids=pillar_ids,
        dry_run=dry_run,
        repo_root=repo_root,
        bundle=bundle,
        use_fakes=use_fakes,
        run_id=run_id,
        enable_hitl=enable_hitl and not dry_run,
        enable_postgres_checkpointer=enable_postgres_checkpointer,
    )
    return _execute(options)


def _execute(opts: RunOptions) -> dict[str, Any]:
    repo_root = opts.repo_root or Path(__file__).resolve().parents[4]
    bundle = opts.bundle or build_default_bundle(
        repo_root, use_fakes=opts.use_fakes or opts.dry_run
    )
    run_id = opts.run_id or uuid.uuid4().hex

    initial: RieState = {
        "run_id": run_id,
        "jurisdiction": opts.jurisdiction,
        "pillar_ids": opts.pillar_ids,
        "dry_run": opts.dry_run,
        "skip_hitl": not opts.enable_hitl,
        "documents": [],
        "raw_bytes_by_doc": {},
        "elements_by_doc": {},
        "edges_by_doc": {},
        "candidate_hits_by_pillar": {},
        "claims": [],
        "verifications": {},
        "coverage": [],
        "errors": [],
        "notes": [],
    }

    tracer = LangfuseTracer.from_env(run_id=run_id, jurisdiction=opts.jurisdiction)
    tracer.start(metadata={"pillar_ids": opts.pillar_ids, "dry_run": opts.dry_run})

    final: RieState
    try:
        try:
            from rie_orchestration.graph import build_graph, build_postgres_checkpointer

            checkpointer = None
            want_pg = (
                opts.enable_postgres_checkpointer
                if opts.enable_postgres_checkpointer is not None
                else os.getenv("RIE_USE_POSTGRES_CHECKPOINTER") == "1"
            )
            if want_pg:
                checkpointer = build_postgres_checkpointer()

            graph = build_graph(
                bundle, checkpointer=checkpointer, with_hitl=opts.enable_hitl
            )
            log.info(
                "run_pipeline: invoking LangGraph (run_id=%s hitl=%s pg=%s)",
                run_id,
                opts.enable_hitl,
                bool(checkpointer),
            )
            config = {"configurable": {"thread_id": run_id}} if checkpointer else {}
            final = graph.invoke(initial, config=config)
        except ImportError:
            log.warning("langgraph not available — sequential fallback")
            final = _sequential_fallback(initial, bundle, with_hitl=opts.enable_hitl)
    finally:
        tracer.stop()

    package = _build_evidence_package(run_id, opts, final)
    package["tracing"] = tracer.summary()
    return package


def _sequential_fallback(
    state: RieState, bundle: AdapterBundle, *, with_hitl: bool
) -> RieState:
    from rie_orchestration.nodes import (
        classify_node,
        coverage_node,
        extract_node,
        index_node,
        ingest_node,
        retrieve_node,
        verify_node,
    )
    from rie_orchestration.review_node import review_node

    nodes = [
        ingest_node,
        extract_node,
        index_node,
        retrieve_node,
        classify_node,
        verify_node,
    ]
    if with_hitl:
        nodes.append(review_node)
    nodes.append(coverage_node)

    for node in nodes:
        state = node(state, bundle)
    return state


def _build_evidence_package(run_id: str, opts: RunOptions, state: RieState) -> dict[str, Any]:
    claims = state.get("claims", [])
    verifications = state.get("verifications", {})
    coverage = state.get("coverage", [])

    package: dict[str, Any] = {
        "run_id": run_id,
        "jurisdiction": opts.jurisdiction,
        "pillar_ids": opts.pillar_ids,
        "finished_at": datetime.now(UTC).isoformat(),
        "counts": {
            "documents": len(state.get("documents", [])),
            "claims": len(claims),
            "verified": sum(
                1 for r in verifications.values() if r.status == VerificationStatus.VERIFIED
            ),
            "flagged": sum(
                1 for r in verifications.values() if r.status == VerificationStatus.FLAGGED
            ),
            "rejected": sum(
                1 for r in verifications.values() if r.status == VerificationStatus.REJECTED
            ),
        },
        "claims": [c.model_dump(mode="json") for c in claims],
        "verifications": {cid: r.model_dump(mode="json") for cid, r in verifications.items()},
        "coverage": [c.model_dump(mode="json") for c in coverage],
    }
    enriched = state.get("enriched_coverage")
    if enriched:
        package["enriched_coverage"] = [
            row.model_dump(mode="json") if hasattr(row, "model_dump") else row
            for row in enriched
        ]
    kg_results = state.get("kg_results")
    if kg_results:
        package["kg_results"] = {
            cid: (r.model_dump(mode="json") if hasattr(r, "model_dump") else r)
            for cid, r in kg_results.items()
        }
    return package
