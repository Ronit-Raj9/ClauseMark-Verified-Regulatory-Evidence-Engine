"""Cost / latency tracking for a pipeline run.

Two pieces:

1. :class:`CostTracker` — a context manager + event recorder. Append events
   with :meth:`CostTracker.record` (or via the convenience helpers
   :meth:`model_call`, :meth:`embed`, :meth:`node`). Get a structured
   :class:`RunCostReport` with :meth:`summarise`.

2. :func:`summarise_run_cost` — the pure function the tracker delegates to.
   Works directly on a list of event dicts so callers that already have
   captured events (e.g. from LangGraph hooks) can summarise without an
   active tracker.

Events
------
Each event is a ``dict`` with keys::

    {"kind": "model_call" | "embed" | "node", "node": str, "seconds": float}

The ``kind`` tells the summariser which counters to bump; the ``node`` is a
free-form label (e.g. ``"classify"`` or ``"verify.entailment"``).
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import TracebackType
from typing import Literal, TypedDict, cast

# ─── Event schema (lightweight) ─────────────────────────────────────────────


EventKind = Literal["model_call", "embed", "node"]


class CostEvent(TypedDict):
    kind: EventKind
    node: str
    seconds: float


# ─── Report ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RunCostReport:
    """Aggregated cost / latency view of a single pipeline run."""

    jurisdiction: str
    pillar_ids: tuple[str, ...]
    total_seconds: float
    model_calls: int
    model_seconds: float
    embedder_calls: int
    claim_count: int
    per_claim_seconds: float
    model_minutes_per_jurisdiction: float


# ─── Pure summariser ────────────────────────────────────────────────────────


def summarise_run_cost(
    events: Sequence[Mapping[str, object]],
    *,
    jurisdiction: str = "",
    pillar_ids: Sequence[str] = (),
    claim_count: int = 0,
    total_seconds: float | None = None,
) -> RunCostReport:
    """Aggregate ``events`` into a :class:`RunCostReport`.

    ``total_seconds`` defaults to the sum of all event seconds; pass an
    explicit value (e.g. wall-clock measured by the tracker) to record the
    true run wall-time rather than the additive per-event total.
    """
    model_calls = 0
    model_seconds = 0.0
    embedder_calls = 0
    summed = 0.0
    for ev in events:
        kind = ev.get("kind")
        raw_secs = ev.get("seconds", 0.0)
        secs = float(cast(float, raw_secs)) if raw_secs is not None else 0.0
        summed += secs
        if kind == "model_call":
            model_calls += 1
            model_seconds += secs
        elif kind == "embed":
            embedder_calls += 1
    total = float(total_seconds) if total_seconds is not None else summed
    per_claim = (total / claim_count) if claim_count > 0 else 0.0
    model_minutes = model_seconds / 60.0
    return RunCostReport(
        jurisdiction=jurisdiction,
        pillar_ids=tuple(pillar_ids),
        total_seconds=total,
        model_calls=model_calls,
        model_seconds=model_seconds,
        embedder_calls=embedder_calls,
        claim_count=claim_count,
        per_claim_seconds=per_claim,
        model_minutes_per_jurisdiction=model_minutes,
    )


# ─── Tracker (context manager) ──────────────────────────────────────────────


@dataclass
class CostTracker:
    """Context manager that records per-event seconds + wall-clock total.

    Usage::

        with CostTracker() as t:
            t.model_call("classify", seconds=0.42)
            t.embed("retrieval", seconds=0.01)
            t.node("verify", seconds=0.13)
        report = t.summarise(
            jurisdiction="SAMPLE", pillar_ids=("6", "7"), claim_count=12
        )

    The tracker captures the wall-clock elapsed between ``__enter__`` and
    ``__exit__`` and feeds it to :func:`summarise_run_cost` as
    ``total_seconds``.
    """

    events: list[CostEvent] = field(default_factory=list)
    _start_monotonic: float | None = None
    _elapsed_seconds: float | None = None

    def __enter__(self) -> CostTracker:
        self._start_monotonic = time.monotonic()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._start_monotonic is not None:
            self._elapsed_seconds = time.monotonic() - self._start_monotonic

    # ─── Recording helpers ──────────────────────────────────────────────

    def record(self, kind: EventKind, node: str, seconds: float) -> None:
        self.events.append({"kind": kind, "node": node, "seconds": float(seconds)})

    def model_call(self, node: str, seconds: float) -> None:
        self.record("model_call", node, seconds)

    def embed(self, node: str, seconds: float) -> None:
        self.record("embed", node, seconds)

    def node(self, node: str, seconds: float) -> None:
        self.record("node", node, seconds)

    # ─── Reporting ──────────────────────────────────────────────────────

    @property
    def elapsed_seconds(self) -> float:
        """Wall-clock seconds between ``__enter__`` and ``__exit__``.

        While the tracker is still active (no ``__exit__`` yet) returns
        the live elapsed reading.
        """
        if self._elapsed_seconds is not None:
            return self._elapsed_seconds
        if self._start_monotonic is not None:
            return time.monotonic() - self._start_monotonic
        return 0.0

    def summarise(
        self,
        jurisdiction: str,
        pillar_ids: Sequence[str],
        claim_count: int,
    ) -> RunCostReport:
        return summarise_run_cost(
            self.events,
            jurisdiction=jurisdiction,
            pillar_ids=pillar_ids,
            claim_count=claim_count,
            total_seconds=self.elapsed_seconds,
        )


__all__ = ["CostEvent", "CostTracker", "EventKind", "RunCostReport", "summarise_run_cost"]
