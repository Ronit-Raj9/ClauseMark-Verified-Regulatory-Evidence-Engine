"""SQLAlchemy engine + session factory."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import Engine
from sqlalchemy import create_engine as _create
from sqlalchemy.orm import Session, sessionmaker


def database_url(override: str | None = None) -> str:
    url = override or os.getenv(
        "DATABASE_URL_SYNC",
        "postgresql+psycopg://rie:rie_dev_password@localhost:5433/rie",
    )
    # Force psycopg (v3) dialect — psycopg2 is not a project dep.
    if url.startswith("postgresql://") and "+psycopg" not in url:
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


def create_engine(url: str | None = None, **kw: Any) -> Engine:
    return _create(database_url(url), pool_pre_ping=True, future=True, **kw)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(engine, expire_on_commit=False, future=True)


@contextmanager
def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
