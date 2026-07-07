"""Source re-fetch — ADVISORY liveness check on the citation's source URL.

A named failure mode in ESCAP's own trials (judges' risk list, see
``rdtii-worked-examples``) is **broken / outdated URLs**: a citation that pointed
to a real instrument at extraction time but whose URL is now dead, moved, or
serving a 4xx/5xx. That does not make the *legal text* wrong, so it must NOT
reject or flag on the required-4 axis — but a reviewer should know the source can
no longer be resolved.

Accordingly this is ADVISORY only:

  * ``verify_source_live`` returns a ``GateResult`` (``gate=ENTITY_GROUNDING``,
    the advisory slot the frozen contract permits) describing whether the URL
    resolves.
  * It lands in ``VerificationReport.advisory_checks`` and NEVER changes
    ``status`` by itself.

Note on the gate name: the frozen ``rie_contracts.GateName`` exposes exactly one
advisory value (``ENTITY_GROUNDING``); all advisory signals re-use it and
disambiguate via ``detail``. The required-4 gate names are reserved.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Final, Protocol, runtime_checkable

import httpx
from rie_contracts import GateName, GateResult

_DEFAULT_TIMEOUT_S: Final[float] = 15.0


@runtime_checkable
class SourceFetcher(Protocol):
    """Re-download the law text at ``source_url``.

    Returns the fetched text, or ``None`` when the URL is dead / unreachable /
    returns a 4xx/5xx. Implementations MUST NOT raise on a dead URL — a dead URL
    is a ``None`` return, which the advisory check renders as a (non-blocking)
    flag.
    """

    def fetch(self, source_url: str) -> str | None: ...


class HttpSourceFetcher:
    """HTTP(S) re-fetcher. ``None`` on any 4xx/5xx, timeout, or transport error."""

    def __init__(self, timeout_s: float = _DEFAULT_TIMEOUT_S) -> None:
        self._timeout_s: float = timeout_s

    @classmethod
    def from_env(cls) -> HttpSourceFetcher:
        timeout_s = float(os.environ.get("RIE_SOURCE_REFETCH_TIMEOUT_S", str(_DEFAULT_TIMEOUT_S)))
        return cls(timeout_s=timeout_s)

    def fetch(self, source_url: str) -> str | None:
        try:
            with httpx.Client(timeout=self._timeout_s, follow_redirects=True) as client:
                response = client.get(source_url)
                response.raise_for_status()
                return response.text
        except (httpx.HTTPError, ValueError):
            # Dead / moved / 4xx / 5xx / timeout — all collapse to "unresolvable".
            return None


class FakeSourceFetcher:
    """Deterministic fetcher for tests — a ``url -> text | None`` mapping.

    A URL absent from the mapping fetches ``None`` (treated as dead).
    """

    def __init__(self, mapping: dict[str, str | None]) -> None:
        self._mapping: dict[str, str | None] = dict(mapping)

    def fetch(self, source_url: str) -> str | None:
        return self._mapping.get(source_url)


def verify_source_live(source_url: str | None, fetcher: SourceFetcher) -> GateResult:
    """ADVISORY: does the citation's ``source_url`` still resolve?

    ``passed=True`` iff the URL is present AND the fetcher returns non-empty text.
    A missing URL, a dead URL, or empty content → ``passed=False`` with a detail
    naming the failure mode (dead / changed URL). This NEVER blocks: it only
    annotates ``advisory_checks`` so a reviewer can act on a broken source.
    """
    now = datetime.now(UTC)

    if not source_url:
        return GateResult(
            gate=GateName.ENTITY_GROUNDING,
            passed=False,
            detail="source re-fetch (advisory): claim has no source_url to verify",
            score=0.0,
            ran_at=now,
        )

    try:
        fetched = fetcher.fetch(source_url)
    except (httpx.HTTPError, RuntimeError, OSError, ValueError) as exc:
        # A fetcher SHOULD return None rather than raise; if it raises anyway,
        # treat it as a dead URL (advisory, never blocks).
        return GateResult(
            gate=GateName.ENTITY_GROUNDING,
            passed=False,
            detail=f"source re-fetch (advisory): fetcher error for {source_url} ({exc})",
            score=0.0,
            ran_at=now,
        )

    if fetched is None or not fetched.strip():
        return GateResult(
            gate=GateName.ENTITY_GROUNDING,
            passed=False,
            detail=(
                f"source re-fetch (advisory): dead/changed URL — {source_url} "
                f"did not resolve to text"
            ),
            score=0.0,
            ran_at=now,
        )

    return GateResult(
        gate=GateName.ENTITY_GROUNDING,
        passed=True,
        detail=f"source re-fetch (advisory): {source_url} resolves",
        score=1.0,
        ran_at=now,
    )


__all__ = [
    "FakeSourceFetcher",
    "HttpSourceFetcher",
    "SourceFetcher",
    "verify_source_live",
]
