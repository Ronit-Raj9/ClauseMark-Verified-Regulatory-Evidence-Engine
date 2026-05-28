"""FastAPI application factory + lifespan.

`app` is the canonical ASGI entrypoint (uvicorn target). The lifespan creates
the live `Settings`, a `ConfigRepository` rooted at the repo, and a session
factory + `DocumentRepository`. The orchestration graph is wired LAZILY —
the API stays importable when `rie_orchestration` (LangGraph + co.) isn't
yet installed in the environment (tests, OpenAPI generation, etc.).
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path


def _load_dotenv() -> None:
    """Bootstrap env from .env at repo root — Makefile `include .env` does not
    propagate to subprocesses, so do it explicitly here."""
    cwd = Path.cwd()
    for candidate in (cwd / ".env", cwd.parent / ".env", cwd.parent.parent / ".env"):
        if candidate.exists():
            for line in candidate.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            return


_load_dotenv()

from fastapi import FastAPI
from rie_config.loader import ConfigRepository
from rie_persistence.models import Base
from rie_persistence.repository import DocumentRepository
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from rie_api.deps import _try_import_concrete_run_graph
from rie_api.middleware import (
    RequestIDMiddleware,
    configure_logging,
    install_exception_handlers,
)
from rie_api.routes import audit, claims, coverage, health, pillars, reviews, runs
from rie_api.settings import Settings, get_settings

logger = logging.getLogger("rie_api.main")


def _build_session_factory(settings: Settings) -> sessionmaker:
    """Build a SQLAlchemy session factory honouring the in-memory toggle.

    In-memory SQLite is the test default; production points at Postgres via
    `database_url_sync` / discrete POSTGRES_* settings.
    """
    if settings.use_in_memory_db:
        engine = create_engine(
            "sqlite:///:memory:",
            future=True,
            connect_args={"check_same_thread": False},
        )
        Base.metadata.create_all(engine)
    else:
        engine = create_engine(settings.sync_dsn(), pool_pre_ping=True, future=True)
    return sessionmaker(engine, expire_on_commit=False, future=True)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)

    # Repositories
    session_factory = _build_session_factory(settings)
    doc_repo = DocumentRepository(session_factory=session_factory)
    config_repo = ConfigRepository(repo_root=settings.repo_root)

    # Orchestration graph (lazy — never block startup on its absence).
    run_graph = _try_import_concrete_run_graph(
        repo_root=settings.repo_root,
        use_fakes=os.getenv("RIE_FORCE_FAKES") == "1",
    )
    if run_graph is None:
        logger.info(
            "rie_orchestration not wired — POST /v1/runs will return 503 "
            "until app.state.run_graph is set"
        )

    app.state.settings = settings
    app.state.session_factory = session_factory
    app.state.doc_repo = doc_repo
    app.state.config_repo = config_repo
    app.state.run_graph = run_graph

    logger.info("rie_api startup complete (env=%s)", settings.env)
    try:
        yield
    finally:
        # Best-effort cleanup of the engine bound to the session factory.
        bind = getattr(session_factory, "kw", {}).get("bind")
        if bind is None:
            bind = session_factory.kw.get("bind") if hasattr(session_factory, "kw") else None
        engine = getattr(doc_repo.session_factory, "bind", None) or bind
        if engine is not None:
            try:
                engine.dispose()
            except Exception:  # pragma: no cover - cleanup best-effort
                logger.warning("engine.dispose failed", exc_info=True)
        logger.info("rie_api shutdown complete")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build a fresh FastAPI app. Useful for tests that override deps.

    When `settings` is provided, the lifespan still runs (and uses
    `get_settings()`), but tests typically bypass lifespan by attaching their
    own state directly to `app.state` and exercising routes via `TestClient`.
    """
    app = FastAPI(
        title="Regulatory Intelligence Engine API",
        version="0.1.0",
        description=(
            "Audit-package + HITL workflow for the RDTII evidence-extraction "
            "and verification engine. Layer-1 facts are deterministic + verified; "
            "Layer-2 scores are recommendations only — reviewers retain final authority."
        ),
        lifespan=lifespan,
    )

    # Middleware
    app.add_middleware(RequestIDMiddleware)
    install_exception_handlers(app)

    # Routers
    app.include_router(health.router)
    app.include_router(pillars.router)
    app.include_router(runs.router)
    app.include_router(claims.router)
    app.include_router(coverage.router)
    app.include_router(audit.router)
    app.include_router(reviews.router)

    return app


app = create_app()
