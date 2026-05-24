"""Typed Streamlit session-state helpers.

We never touch ``st.session_state`` directly from page code — every read or
write goes through one of these helpers. That keeps the keyspace small, the
types explicit, and the pages testable in isolation.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, cast

from rie_contracts.models import Layer1Status

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass


class _StateLike(Protocol):
    """Minimal protocol satisfied by ``st.session_state`` and ``dict``."""

    def __getitem__(self, key: str) -> Any: ...
    def __setitem__(self, key: str, value: Any) -> None: ...
    def __contains__(self, key: object) -> bool: ...
    def get(self, key: str, default: Any = None) -> Any: ...


# Session-state key namespace — keep this short, prefix all keys.
_K_FILTER = "rie.claims.filter"
_K_ACTIVE_CLAIM = "rie.claims.active_id"
_K_ACTIVE_RUN = "rie.runs.active_id"
_K_JURISDICTION = "rie.global.jurisdiction"
_K_REVIEWER = "rie.global.reviewer"


@dataclass(slots=True)
class ClaimsFilter:
    """Filter state for the Claims table."""

    jurisdiction: str | None = None
    pillar_id: str | None = None
    indicator_id: str | None = None
    status: Layer1Status | None = None
    search: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "jurisdiction": self.jurisdiction,
            "pillar_id": self.pillar_id,
            "indicator_id": self.indicator_id,
            "status": self.status.value if self.status is not None else None,
            "search": self.search,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> ClaimsFilter:
        if not data:
            return cls()
        raw_status = data.get("status")
        status = Layer1Status(raw_status) if raw_status else None
        return cls(
            jurisdiction=data.get("jurisdiction") or None,
            pillar_id=data.get("pillar_id") or None,
            indicator_id=data.get("indicator_id") or None,
            status=status,
            search=str(data.get("search") or ""),
        )


def _store(state: _StateLike | None) -> _StateLike:
    if state is not None:
        return state
    # Lazy import keeps non-streamlit callers (tests, type-checkers) happy.
    import streamlit as st

    return cast(_StateLike, st.session_state)


# ──────────────────────────────────────────────────────────────────────
# Filter
# ──────────────────────────────────────────────────────────────────────


def get_filter(state: _StateLike | None = None) -> ClaimsFilter:
    store = _store(state)
    raw = store.get(_K_FILTER)
    if isinstance(raw, ClaimsFilter):
        return raw
    if isinstance(raw, Mapping):
        return ClaimsFilter.from_mapping(raw)
    return ClaimsFilter()


def set_filter(value: ClaimsFilter, state: _StateLike | None = None) -> None:
    _store(state)[_K_FILTER] = value


def clear_filter(state: _StateLike | None = None) -> None:
    _store(state)[_K_FILTER] = ClaimsFilter()


# ──────────────────────────────────────────────────────────────────────
# Active claim selection
# ──────────────────────────────────────────────────────────────────────


def get_active_claim(state: _StateLike | None = None) -> str | None:
    raw = _store(state).get(_K_ACTIVE_CLAIM)
    return str(raw) if raw else None


def set_active_claim(claim_id: str | None, state: _StateLike | None = None) -> None:
    _store(state)[_K_ACTIVE_CLAIM] = claim_id


# ──────────────────────────────────────────────────────────────────────
# Active run selection
# ──────────────────────────────────────────────────────────────────────


def get_active_run(state: _StateLike | None = None) -> str | None:
    raw = _store(state).get(_K_ACTIVE_RUN)
    return str(raw) if raw else None


def set_active_run(run_id: str | None, state: _StateLike | None = None) -> None:
    _store(state)[_K_ACTIVE_RUN] = run_id


# ──────────────────────────────────────────────────────────────────────
# Global jurisdiction / reviewer
# ──────────────────────────────────────────────────────────────────────


def get_jurisdiction(state: _StateLike | None = None) -> str | None:
    raw = _store(state).get(_K_JURISDICTION)
    return str(raw) if raw else None


def set_jurisdiction(value: str | None, state: _StateLike | None = None) -> None:
    _store(state)[_K_JURISDICTION] = value


def get_reviewer(state: _StateLike | None = None) -> str:
    raw = _store(state).get(_K_REVIEWER)
    return str(raw) if raw else "anonymous"


def set_reviewer(value: str, state: _StateLike | None = None) -> None:
    _store(state)[_K_REVIEWER] = value


@dataclass(slots=True)
class _Sentinels:
    """Exposed only for testing — keys used by this module."""

    filter: str = _K_FILTER
    active_claim: str = _K_ACTIVE_CLAIM
    active_run: str = _K_ACTIVE_RUN
    jurisdiction: str = _K_JURISDICTION
    reviewer: str = _K_REVIEWER


SESSION_KEYS = _Sentinels()


__all__ = [
    "SESSION_KEYS",
    "ClaimsFilter",
    "clear_filter",
    "get_active_claim",
    "get_active_run",
    "get_filter",
    "get_jurisdiction",
    "get_reviewer",
    "set_active_claim",
    "set_active_run",
    "set_filter",
    "set_jurisdiction",
    "set_reviewer",
]


# Silence the unused-import warning for MutableMapping; it is part of the
# public-ish type surface for future helpers.
_ = MutableMapping
_ = field
