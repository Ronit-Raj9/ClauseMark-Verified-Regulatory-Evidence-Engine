"""LangGraph graph construction. Nodes wired in dependency order with HITL.

The graph follows §4 of the architecture:
  ingest → extract → index → retrieve → classify → verify → review (interrupt for FLAGGED) → coverage [→ layer2]
"""

from __future__ import annotations

import logging
from collections.abc import Callable

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
from rie_orchestration.state import RieState
from rie_orchestration.tracing import LangfuseTracer
from rie_orchestration.wiring import AdapterBundle

log = logging.getLogger(__name__)

_MEMORY_SAVER: object | None = None


def build_graph(
    bundle: AdapterBundle,
    *,
    checkpointer: object | None = None,
    with_hitl: bool = True,
    tracer: LangfuseTracer | None = None,
):
    """Compile the LangGraph.

    `with_hitl=False` skips the review node (used for dry-runs / tests where no
    checkpointer is wired so interrupts cannot resume).
    """
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as e:
        raise ImportError(
            "langgraph required. uv add --package rie-orchestration langgraph"
        ) from e

    graph = StateGraph(RieState)

    def _bind(
        name: str,
        fn: Callable[[RieState, AdapterBundle], RieState],
    ) -> Callable[[RieState], RieState]:
        def _run(state: RieState) -> RieState:
            if tracer is not None:
                with tracer.span(name):
                    return fn(state, bundle)
            return fn(state, bundle)

        return _run

    graph.add_node("ingest", _bind("ingest", ingest_node))
    graph.add_node("extract", _bind("extract", extract_node))
    graph.add_node("index", _bind("index", index_node))
    graph.add_node("retrieve", _bind("retrieve", retrieve_node))
    graph.add_node("classify", _bind("classify", classify_node))
    graph.add_node("verify", _bind("verify", verify_node))
    if with_hitl:
        graph.add_node("review", _bind("review", review_node))
    graph.add_node("coverage", _bind("coverage", coverage_node))
    graph.add_node("layer2", _bind("layer2", layer2_node))

    graph.add_edge(START, "ingest")
    graph.add_edge("ingest", "extract")
    graph.add_edge("extract", "index")
    graph.add_edge("index", "retrieve")
    graph.add_edge("retrieve", "classify")
    graph.add_edge("classify", "verify")
    if with_hitl:
        graph.add_edge("verify", "review")
        graph.add_edge("review", "coverage")
    else:
        graph.add_edge("verify", "coverage")
    graph.add_edge("coverage", "layer2")
    graph.add_edge("layer2", END)

    kwargs: dict[str, object] = {}
    if checkpointer is not None:
        kwargs["checkpointer"] = checkpointer
    return graph.compile(**kwargs)


def build_memory_checkpointer(*, reset: bool = False) -> object | None:
    """Return a process-wide in-memory checkpointer for HITL resume in dev/tests."""
    global _MEMORY_SAVER
    if reset:
        _MEMORY_SAVER = None
    if _MEMORY_SAVER is not None:
        return _MEMORY_SAVER
    try:
        from langgraph.checkpoint.memory import MemorySaver

        _MEMORY_SAVER = MemorySaver()
        return _MEMORY_SAVER
    except ImportError:
        log.warning("langgraph checkpoint memory saver unavailable")
        return None


def build_postgres_checkpointer(dsn: str | None = None) -> object | None:
    """Return `PostgresSaver` if `langgraph-checkpoint-postgres` installed, else None.

    Caller passes it (or not) to `build_graph(..., checkpointer=...)`.
    """
    import os

    dsn = dsn or os.getenv("DATABASE_URL_SYNC")
    if not dsn:
        return None
    try:
        from langgraph.checkpoint.postgres import PostgresSaver

        saver = PostgresSaver.from_conn_string(dsn)
        saver.setup()
        return saver
    except ImportError:
        log.warning("langgraph-checkpoint-postgres not installed; in-memory only")
        return None
    except Exception as e:
        log.warning("PostgresSaver init failed (%s); in-memory only", e)
        return None
