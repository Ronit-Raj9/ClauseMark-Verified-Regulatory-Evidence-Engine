"""Postgres persistence — repos behind DocumentRepositoryPort."""

from rie_persistence.db import create_engine, session_scope
from rie_persistence.repository import DocumentRepository

__all__ = ["DocumentRepository", "create_engine", "session_scope"]
