"""Authoritative-source reachability — the §7 *honest-absence* upgrade (Phase 2).

``systemArchitecture.md`` §1 lists "defensible absence scoring beyond the
provided corpus" as Phase 2 roadmap, and §7 frames absence as a *measured,
bounded* statement rather than a silent ``0``. This module strengthens the
``no_evidence_in_searched_corpus`` → defensible-real-``0`` boundary by checking
two independent signals:

1. **Gold-set recall** — how much of the *known* corpus we actually retrieved.
2. **Authoritative-source reachability** — what fraction of the source TYPES we
   expect for this ``(indicator, jurisdiction)`` we were actually able to reach.

A ``no_evidence`` only becomes a *candidate* for a defensible real ``0`` when
BOTH signals clear their floors. Crucially, this module NEVER emits a ``0`` and
NEVER changes the coverage state: it only produces a verdict a human reviewer
MAY use to justify treating the result as a real ``0``. The bare ``0`` remains
structurally impossible — exactly as §7 demands.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rie_contracts import CoverageState

__all__ = [
    "AuthoritativeSourceCheck",
    "assess_reachability",
    "is_defensible_zero",
]


def assess_reachability(expected: list[str], reached: list[str]) -> float:
    """Fraction of expected authoritative-source TYPES we were able to reach.

    Reachability is set-based: it is the fraction of DISTINCT expected source
    types that appear in ``reached``. Duplicate or unexpected entries in
    ``reached`` do not inflate the score.

    * ``expected=[]`` → ``1.0`` — nothing was expected, so nothing is missing
      (vacuously fully reachable).
    * otherwise → ``len(set(expected) & set(reached)) / len(set(expected))``.
    """
    expected_set = set(expected)
    if not expected_set:
        return 1.0
    reached_set = set(reached)
    hit = len(expected_set & reached_set)
    return hit / len(expected_set)


def is_defensible_zero(
    coverage_state: CoverageState,
    reachability: float,
    gold_recall: float | None,
    *,
    recall_floor: float = 0.8,
    reach_floor: float = 0.9,
) -> bool:
    """Return True iff a ``no_evidence`` result MAY be treated as a real ``0``.

    This is the §7 honest-absence upgrade. It does NOT emit a ``0`` and does NOT
    mutate the coverage state — it only tells a reviewer whether the absence is
    *defensible* as a real ``0`` because the corpus was both well-covered
    (``gold_recall >= recall_floor``) and authoritatively complete
    (``reachability >= reach_floor``).

    Returns True ONLY when ALL of the following hold:

    * ``coverage_state == NO_EVIDENCE_IN_SEARCHED_CORPUS`` — a defensible ``0``
      is only meaningful when we actually searched and found nothing. An
      ``EVIDENCE_FOUND`` already has evidence; an ``INSUFFICIENT_COVERAGE`` has
      no measured recall to bound the miss risk.
    * ``gold_recall is not None and gold_recall >= recall_floor``.
    * ``reachability >= reach_floor``.

    In every other case the result stays a bounded "no evidence in searched
    corpus" statement. The bare ``0`` is STILL never auto-emitted; this flag
    only marks where a reviewer MAY exercise that judgement.
    """
    if coverage_state != CoverageState.NO_EVIDENCE_IN_SEARCHED_CORPUS:
        return False
    if gold_recall is None or gold_recall < recall_floor:
        return False
    return reachability >= reach_floor


@dataclass(frozen=True)
class AuthoritativeSourceCheck:
    """Record of an authoritative-source reachability assessment.

    Pure data: ``reachability`` is derived from ``sources_reached`` /
    ``sources_total`` if not supplied, keeping the two consistent by default.
    """

    indicator_id: str
    jurisdiction: str
    expected_source_types: list[str] = field(default_factory=list)
    sources_reached: int = 0
    sources_total: int = 0
    reachability: float = 0.0

    def __post_init__(self) -> None:
        # Derive reachability from the counts when the caller leaves it at the
        # default 0.0 but there *are* sources to count — keeps the dataclass
        # internally consistent without forcing every caller to pre-compute it.
        if self.reachability == 0.0 and self.sources_total > 0:
            object.__setattr__(self, "reachability", self.sources_reached / self.sources_total)
