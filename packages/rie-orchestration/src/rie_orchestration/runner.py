"""High-level runner — the entry point used by the CLI + the API."""

from __future__ import annotations

import logging
import os
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rie_contracts import VerificationStatus
from rie_orchestration.citations import materialise_citations
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


def resume_pipeline(
    run_id: str,
    decisions: dict[str, str],
    *,
    repo_root: Path | None = None,
    bundle: AdapterBundle | None = None,
    use_fakes: bool = False,
    enable_postgres_checkpointer: bool | None = None,
) -> dict[str, Any]:
    """Resume a HITL-interrupted graph with reviewer decisions.

    ``decisions`` maps ``claim_id`` → ``accept`` | ``correct`` | ``reject`` and
    is passed to LangGraph via ``Command(resume=...)``.
    """
    repo_root = repo_root or Path(__file__).resolve().parents[4]
    bundle = bundle or build_default_bundle(repo_root, use_fakes=use_fakes)
    checkpointer = _resolve_checkpointer(
        enable_hitl=True,
        enable_postgres_checkpointer=enable_postgres_checkpointer,
    )
    if checkpointer is None:
        msg = "resume_pipeline requires a checkpointer (Postgres or in-memory)"
        raise RuntimeError(msg)

    try:
        from langgraph.types import Command  # noqa: F401 — re-export path for resume
    except ImportError as e:
        raise ImportError("langgraph required for resume_pipeline") from e

    from rie_orchestration.graph import build_graph

    tracer = LangfuseTracer.from_env(run_id=run_id)
    tracer.start(metadata={"resume": True, "decisions": list(decisions.keys())})

    graph = build_graph(bundle, checkpointer=checkpointer, with_hitl=True, tracer=tracer)
    config = {"configurable": {"thread_id": run_id}}
    snapshot = _graph_snapshot(graph, config)
    if not _graph_has_interrupt(snapshot):
        msg = f"run {run_id!r} has no pending HITL interrupt to resume"
        raise RuntimeError(msg)
    resume_cmd = _build_resume_command(snapshot, decisions)
    try:
        final = graph.invoke(resume_cmd, config=config)
    finally:
        tracer.stop()

    snapshot = _graph_snapshot(graph, config)
    jurisdiction = str(final.get("jurisdiction", ""))
    pillar_ids = list(final.get("pillar_ids", []))
    opts = RunOptions(
        jurisdiction=jurisdiction,
        pillar_ids=pillar_ids,
        repo_root=repo_root,
        bundle=bundle,
        run_id=run_id,
    )
    package = _build_evidence_package(run_id, opts, final, bundle=bundle)
    package["tracing"] = tracer.summary()
    package["status"] = _run_status(final, snapshot)
    package["detail"] = _run_detail(final, snapshot)
    return package


def build_run_graph(
    *,
    repo_root: Path | None = None,
    use_fakes: bool | None = None,
) -> Callable[[Any], dict[str, str]]:
    """Return the API ``RunGraph`` callable (``RunRequest`` → response dict).

    The API lazy-imports this factory; the returned callable accepts any object
    with ``jurisdiction``, ``pillar_ids``, and optional ``run_id`` attributes.
    """
    root = repo_root or Path(__file__).resolve().parents[4]
    force_fakes = (
        use_fakes if use_fakes is not None else os.getenv("RIE_FORCE_FAKES") == "1"
    )
    bundle = build_default_bundle(root, use_fakes=force_fakes)

    def _run(req: Any) -> dict[str, str]:
        jurisdiction = str(getattr(req, "jurisdiction", "") or "")
        pillar_ids = list(getattr(req, "pillar_ids", []) or [])
        run_id = getattr(req, "run_id", None) or uuid.uuid4().hex
        package = run_pipeline(
            jurisdiction=jurisdiction,
            pillar_ids=pillar_ids,
            run_id=run_id,
            repo_root=root,
            bundle=bundle,
            use_fakes=force_fakes,
        )
        return {
            "run_id": package["run_id"],
            "status": str(package.get("status", "completed")),
            "detail": str(package.get("detail", f"processed {len(pillar_ids)} pillar(s)")),
        }

    return _run


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
    snapshot: object | None = None
    graph: object | None = None
    config: dict[str, object] = {}
    try:
        try:
            from rie_orchestration.graph import build_graph

            checkpointer = _resolve_checkpointer(
                enable_hitl=opts.enable_hitl,
                enable_postgres_checkpointer=opts.enable_postgres_checkpointer,
            )
            graph = build_graph(
                bundle,
                checkpointer=checkpointer,
                with_hitl=opts.enable_hitl,
                tracer=tracer,
            )
            log.info(
                "run_pipeline: invoking LangGraph (run_id=%s hitl=%s cp=%s)",
                run_id,
                opts.enable_hitl,
                type(checkpointer).__name__ if checkpointer else None,
            )
            if checkpointer is not None:
                config = {"configurable": {"thread_id": run_id}}
            final = graph.invoke(initial, config=config)
            snapshot = _graph_snapshot(graph, config) if config else None
        except ImportError:
            log.warning("langgraph not available — sequential fallback")
            final = _sequential_fallback(
                initial, bundle, with_hitl=opts.enable_hitl, tracer=tracer
            )
    finally:
        tracer.stop()

    package = _build_evidence_package(run_id, opts, final, bundle=bundle)
    package["tracing"] = tracer.summary()
    package["status"] = _run_status(final, snapshot)
    package["detail"] = _run_detail(final, snapshot)
    return package


def _resolve_checkpointer(
    *,
    enable_hitl: bool,
    enable_postgres_checkpointer: bool | None,
) -> object | None:
    from rie_orchestration.graph import build_memory_checkpointer, build_postgres_checkpointer

    want_pg = (
        enable_postgres_checkpointer
        if enable_postgres_checkpointer is not None
        else os.getenv("RIE_USE_POSTGRES_CHECKPOINTER") == "1"
    )
    if want_pg:
        pg = build_postgres_checkpointer()
        if pg is not None:
            return pg
    if enable_hitl:
        return build_memory_checkpointer()
    return None


def _graph_snapshot(graph: object, config: dict[str, object]) -> object | None:
    getter = getattr(graph, "get_state", None)
    if not callable(getter) or not config:
        return None
    try:
        return getter(config)
    except Exception:
        log.debug("graph.get_state failed", exc_info=True)
        return None


def _graph_has_interrupt(snapshot: object | None) -> bool:
    if snapshot is None:
        return False
    tasks = getattr(snapshot, "tasks", None) or ()
    for task in tasks:
        interrupts = getattr(task, "interrupts", None) or ()
        if interrupts:
            return True
    return False


def _interrupt_ids(snapshot: object) -> list[str]:
    ids: list[str] = []
    for task in getattr(snapshot, "tasks", None) or ():
        for intr in getattr(task, "interrupts", None) or ():
            interrupt_id = getattr(intr, "id", None)
            if isinstance(interrupt_id, str):
                ids.append(interrupt_id)
    return ids


def _build_resume_command(snapshot: object, decisions: dict[str, str]) -> object:
    from langgraph.types import Command

    interrupt_ids = _interrupt_ids(snapshot)
    if not interrupt_ids:
        msg = "cannot resume — no interrupt ids on graph snapshot"
        raise RuntimeError(msg)
    if len(interrupt_ids) == 1:
        return Command(resume={interrupt_ids[0]: decisions})
    return Command(resume={interrupt_id: decisions for interrupt_id in interrupt_ids})


def _run_status(final: RieState, snapshot: object | None) -> str:
    if _graph_has_interrupt(snapshot):
        return "running"
    if final.get("errors"):
        return "failed"
    return "completed"


def _run_detail(final: RieState, snapshot: object | None) -> str:
    if _graph_has_interrupt(snapshot):
        flagged = sum(
            1
            for r in final.get("verifications", {}).values()
            if r.status == VerificationStatus.FLAGGED
        )
        return f"awaiting human review for {flagged} flagged claim(s)"
    claim_count = len(final.get("claims", []))
    return f"processed run with {claim_count} claim(s)"


def _sequential_fallback(
    state: RieState,
    bundle: AdapterBundle,
    *,
    with_hitl: bool,
    tracer: LangfuseTracer | None = None,
) -> RieState:
    from rie_orchestration.nodes import (
        classify_node,
        coverage_node,
        extract_node,
        index_node,
        ingest_node,
        layer2_node,
        retrieve_node,
        verify_node,
    )
    from rie_orchestration.review_node import review_node

    steps: list[tuple[str, object]] = [
        ("ingest", ingest_node),
        ("extract", extract_node),
        ("index", index_node),
        ("retrieve", retrieve_node),
        ("classify", classify_node),
        ("verify", verify_node),
    ]
    if with_hitl:
        steps.append(("review", review_node))
    steps.append(("coverage", coverage_node))
    steps.append(("layer2", layer2_node))

    for name, node in steps:
        if tracer is not None:
            with tracer.span(name):
                state = node(state, bundle)  # type: ignore[operator]
        else:
            state = node(state, bundle)  # type: ignore[operator]
    return state


def _build_evidence_package(
    run_id: str,
    opts: RunOptions,
    state: RieState,
    *,
    bundle: AdapterBundle | None = None,
) -> dict[str, Any]:
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
    layer2 = state.get("layer2_recommendations", [])
    package["layer2_recommendations"] = [
        r.model_dump(mode="json") if hasattr(r, "model_dump") else r for r in layer2
    ]

    if bundle is not None:
        citations_by_claim = _materialise_claim_citations(claims, bundle.repo)
        if citations_by_claim:
            package["citations"] = citations_by_claim

    return package


def _materialise_claim_citations(
    claims: list[Any],
    repo: object,
) -> dict[str, list[dict[str, object]]]:
    get_document = getattr(repo, "get_document", None)
    if not callable(get_document):
        return {}
    out: dict[str, list[dict[str, object]]] = {}
    for claim in claims:
        built = materialise_citations(
            claim.evidence_spans,
            get_element_text=repo.get_element_text,  # type: ignore[attr-defined]
            get_element=repo.get_element,  # type: ignore[attr-defined]
            get_document=get_document,
        )
        if built:
            out[claim.claim_id] = [
                {
                    "span_id": c.span_id,
                    "doc_id": c.doc_id,
                    "document_title": c.document_title,
                    "legal_locator": c.legal_locator,
                    "snippet": c.snippet,
                    "char_start": c.char_start,
                    "char_end": c.char_end,
                }
                for c in built
            ]
    return out
