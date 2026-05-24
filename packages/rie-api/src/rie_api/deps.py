"""FastAPI dependency providers — the seam between routes and adapters.

Everything routers need is fetched through these getters. No module-level
state: the live `Settings`, repositories, and graph runner are attached to
`app.state` during the lifespan event and retrieved per-request via `Request`.

`get_run_graph` performs a **lazy** import of `rie_orchestration` — the API
must remain importable (for tests + OpenAPI generation) even when LangGraph
isn't yet wired or its heavy deps aren't installed in the test env.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from fastapi import Depends, Request
from rie_config.loader import ConfigRepository
from rie_persistence.repository import DocumentRepository

from rie_api.errors import DependencyUnavailableError
from rie_api.schemas import RunRequest, RunResponse
from rie_api.settings import Settings

logger = logging.getLogger("rie_api.deps")


class RunGraph(Protocol):
    """Callable that executes the orchestration pipeline for one run.

    The concrete implementation lives in `rie_orchestration` and is imported
    lazily; this Protocol is the only thing this package types against.
    """

    def __call__(self, req: RunRequest) -> RunResponse: ...


# ──────────────────────────────────────────────────────────────────────────
# Settings
# ──────────────────────────────────────────────────────────────────────────


def get_settings(request: Request) -> Settings:
    settings: Settings | None = getattr(request.app.state, "settings", None)
    if settings is None:
        raise DependencyUnavailableError("settings not initialised on app.state")
    return settings


# ──────────────────────────────────────────────────────────────────────────
# Repositories
# ──────────────────────────────────────────────────────────────────────────


def get_repo(request: Request) -> DocumentRepository:
    repo: DocumentRepository | None = getattr(request.app.state, "doc_repo", None)
    if repo is None:
        raise DependencyUnavailableError("document repository not initialised")
    return repo


def get_config_repo(request: Request) -> ConfigRepository:
    repo: ConfigRepository | None = getattr(request.app.state, "config_repo", None)
    if repo is None:
        raise DependencyUnavailableError("config repository not initialised")
    return repo


# ──────────────────────────────────────────────────────────────────────────
# Orchestration (lazy)
# ──────────────────────────────────────────────────────────────────────────


def _try_import_concrete_run_graph() -> RunGraph | None:
    """Best-effort import of the concrete graph runner from `rie_orchestration`.

    Returns ``None`` if the package isn't installed or doesn't yet expose
    `build_run_graph` — the route surfaces a 503 in that case.
    """
    try:
        import rie_orchestration as _orch  # noqa: PLC0415  (lazy)
    except Exception as exc:  # pragma: no cover - depends on optional install
        logger.warning("rie_orchestration import failed: %s", exc)
        return None

    builder: Callable[..., Any] | None = getattr(_orch, "build_run_graph", None)
    if builder is None:
        logger.info("rie_orchestration present but build_run_graph not yet exposed")
        return None
    try:
        graph = builder()
    except Exception as exc:  # pragma: no cover - construction failure
        logger.warning("build_run_graph() raised: %s", exc)
        return None

    if not callable(graph):
        return None
    return graph  # type: ignore[return-value]


def get_run_graph(request: Request) -> RunGraph:
    """Return the wired `RunGraph` callable, raising 503 if unavailable.

    The app's lifespan stashes a graph on `app.state.run_graph`. If nothing
    was set (e.g. tests didn't override, orchestration not installed), we
    attempt a lazy import as a last resort.
    """
    graph: RunGraph | None = getattr(request.app.state, "run_graph", None)
    if graph is not None:
        return graph
    graph = _try_import_concrete_run_graph()
    if graph is None:
        raise DependencyUnavailableError(
            "orchestration graph not available — wire app.state.run_graph "
            "or install rie_orchestration"
        )
    request.app.state.run_graph = graph
    return graph


# ──────────────────────────────────────────────────────────────────────────
# Convenience aliases used by routers
# ──────────────────────────────────────────────────────────────────────────

SettingsDep = Depends(get_settings)
RepoDep = Depends(get_repo)
ConfigRepoDep = Depends(get_config_repo)
RunGraphDep = Depends(get_run_graph)


__all__: Sequence[str] = (
    "ConfigRepoDep",
    "RepoDep",
    "RunGraph",
    "RunGraphDep",
    "SettingsDep",
    "get_config_repo",
    "get_repo",
    "get_run_graph",
    "get_settings",
)
