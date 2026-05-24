"""LangGraph graph construction. Nodes wired in dependency order with HITL.

The graph follows §4 of the architecture:
  ingest → extract → index → retrieve → classify → verify → review (interrupt for FLAGGED) → coverage
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
    retrieve_node,
    verify_node,
)
from rie_orchestration.review_node import review_node
from rie_orchestration.state import RieState
from rie_orchestration.wiring import AdapterBundle

log = logging.getLogger(__name__)


def build_graph(
    bundle: AdapterBundle,
    *,
    checkpointer: object | None = None,
    with_hitl: bool = True,
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
        fn: Callable[[RieState, AdapterBundle], RieState],
    ) -> Callable[[RieState], RieState]:
        return lambda s: fn(s, bundle)

    graph.add_node("ingest", _bind(ingest_node))
    graph.add_node("extract", _bind(extract_node))
    graph.add_node("index", _bind(index_node))
    graph.add_node("retrieve", _bind(retrieve_node))
    graph.add_node("classify", _bind(classify_node))
    graph.add_node("verify", _bind(verify_node))
    if with_hitl:
        graph.add_node("review", _bind(review_node))
    graph.add_node("coverage", _bind(coverage_node))

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
    graph.add_edge("coverage", END)

    kwargs: dict[str, object] = {}
    if checkpointer is not None:
        kwargs["checkpointer"] = checkpointer
    return graph.compile(**kwargs)


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
