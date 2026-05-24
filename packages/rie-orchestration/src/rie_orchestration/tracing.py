"""Optional Langfuse tracing wrapper.

Activated when LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY env vars set. Falls
back to a no-op `Tracer` if `langfuse` is not installed or keys absent — runs
never fail because tracing fails.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class LangfuseTracer:
    """Lightweight wrapper around langfuse.Trace.

    - `start(metadata)` opens the trace.
    - `span(name)` returns a context manager recording (node, seconds).
    - `stop()` flushes the trace.
    - `summary()` returns a serialisable dict of recorded spans + totals (used
      by tests / the evidence package's tracing footer even if Langfuse is
      not configured).
    """

    run_id: str
    jurisdiction: str | None = None
    host: str | None = None
    public_key: str | None = None
    secret_key: str | None = None
    enabled: bool = False
    _langfuse: Any = None
    _trace: Any = None
    _events: list[dict[str, Any]] = field(default_factory=list)
    _t0: float = 0.0

    @classmethod
    def from_env(cls, *, run_id: str, jurisdiction: str | None = None) -> LangfuseTracer:
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY") or None
        secret_key = os.getenv("LANGFUSE_SECRET_KEY") or None
        host = os.getenv("LANGFUSE_HOST") or None
        enabled = bool(public_key and secret_key)
        return cls(
            run_id=run_id,
            jurisdiction=jurisdiction,
            host=host,
            public_key=public_key,
            secret_key=secret_key,
            enabled=enabled,
        )

    def start(self, *, metadata: dict[str, Any] | None = None) -> None:
        self._t0 = time.monotonic()
        if not self.enabled:
            return
        try:
            from langfuse import Langfuse

            self._langfuse = Langfuse(
                public_key=self.public_key,
                secret_key=self.secret_key,
                host=self.host or "https://cloud.langfuse.com",
            )
            self._trace = self._langfuse.trace(
                id=self.run_id,
                name="rie_pipeline",
                metadata={
                    "jurisdiction": self.jurisdiction,
                    **(metadata or {}),
                },
            )
        except Exception as e:
            log.warning("Langfuse init failed (%s) — tracing disabled", e)
            self.enabled = False

    def span(self, name: str):
        return _SpanContext(self, name)

    def record(self, name: str, seconds: float, **kw: Any) -> None:
        evt = {"name": name, "seconds": seconds, **kw}
        self._events.append(evt)
        if self.enabled and self._trace is not None:
            try:
                self._trace.span(name=name, metadata={"seconds": seconds, **kw})
            except Exception as e:
                log.debug("langfuse span record failed: %s", e)

    def stop(self) -> None:
        if not self.enabled or self._langfuse is None:
            return
        try:
            self._langfuse.flush()
        except Exception as e:
            log.warning("langfuse flush failed: %s", e)

    def summary(self) -> dict[str, Any]:
        total = sum(e.get("seconds", 0.0) for e in self._events)
        return {
            "enabled": self.enabled,
            "run_id": self.run_id,
            "total_seconds": round(total, 4),
            "events": list(self._events),
            "wall_seconds": round(time.monotonic() - self._t0, 4) if self._t0 else 0.0,
        }


class _SpanContext:
    def __init__(self, tracer: LangfuseTracer, name: str):
        self._tracer = tracer
        self._name = name
        self._t0 = 0.0

    def __enter__(self) -> _SpanContext:
        self._t0 = time.monotonic()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        elapsed = time.monotonic() - self._t0
        self._tracer.record(self._name, elapsed, exc=str(exc_type) if exc_type else None)
