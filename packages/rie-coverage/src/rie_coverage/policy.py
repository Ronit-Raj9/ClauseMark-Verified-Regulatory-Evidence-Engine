"""Pure-function helpers documenting the 3-state coverage policy.

These functions contain ZERO I/O and ZERO model calls. They are the canonical
specification of the absence-reasoning policy described in
``systemArchitecture.md`` §7. The service layer (:mod:`rie_coverage.service`)
composes them into a port implementation.

Design contract — restated so a reviewer can audit it from one file:

1. A bare ``0`` is NEVER produced. There are exactly three coverage states.
2. ``EVIDENCE_FOUND`` requires at least one VERIFIED claim that matches BOTH
   the requested ``indicator_id`` AND ``jurisdiction``. Pending / flagged /
   rejected claims do not count.
3. ``NO_EVIDENCE_IN_SEARCHED_CORPUS`` requires a measured gold-set recall.
   Without recall we cannot bound the miss risk, so the honest output is
   ``INSUFFICIENT_COVERAGE`` — not a silent zero.
4. ``INSUFFICIENT_COVERAGE`` is the safe-by-default state. It is also the
   trigger to ask the operator to widen the source corpus.
"""

from __future__ import annotations

from collections.abc import Sequence

from rie_contracts import Claim, CoverageState, Layer1Status

# ────────────────────────────────────────────────────────────────────────────
# Public policy primitives
# ────────────────────────────────────────────────────────────────────────────


def filter_qualifying_claims(
    verified_claims: Sequence[Claim],
    indicator_id: str,
    jurisdiction: str,
) -> list[Claim]:
    """Return only claims that count as evidence for this (jurisdiction, indicator).

    Filtering rules (all must hold):

    * ``layer1_status == Layer1Status.VERIFIED`` — pending / flagged / rejected
      claims are NEVER counted as evidence.
    * ``indicator_id`` matches exactly.
    * ``jurisdiction`` matches exactly.
    """
    out: list[Claim] = []
    for c in verified_claims:
        if c.layer1_status != Layer1Status.VERIFIED:
            continue
        if c.indicator_id != indicator_id:
            continue
        if c.jurisdiction != jurisdiction:
            continue
        out.append(c)
    return out


def classify(
    verified_claims: Sequence[Claim],
    gold_recall: float | None,
    indicator_id: str,
    jurisdiction: str,
) -> CoverageState:
    """Return the coverage state for one ``(jurisdiction, indicator)`` pair.

    A bare ``0`` is never produced — exactly one of three states is returned.
    """
    qualifying = filter_qualifying_claims(verified_claims, indicator_id, jurisdiction)
    if qualifying:
        return CoverageState.EVIDENCE_FOUND
    if gold_recall is None:
        return CoverageState.INSUFFICIENT_COVERAGE
    return CoverageState.NO_EVIDENCE_IN_SEARCHED_CORPUS


def miss_pct_from_recall(gold_recall: float) -> int:
    """Convert a recall in ``[0.0, 1.0]`` to an integer miss-risk percentage.

    Rounding rule: ``int(round((1 - gold_recall) * 100))`` using Python's
    built-in ``round``, which is banker's rounding (round-half-to-even).
    In practice, because most recalls do not produce a mathematically exact
    ``*.5`` after float multiplication, the observable behaviour is "nearest
    whole percent." When an exact half *is* produced, ties break toward the
    even integer.

    Examples (verified in the test suite):

    * ``recall=1.0`` → miss_pct=0
    * ``recall=0.99`` → miss_pct=1
    * ``recall=0.82`` → miss_pct=18
    * ``recall=0.5``  → miss_pct=50
    * ``recall=0.0``  → miss_pct=100
    """
    if not 0.0 <= gold_recall <= 1.0:
        msg = f"gold_recall must be in [0.0, 1.0], got {gold_recall}"
        raise ValueError(msg)
    miss = (1.0 - gold_recall) * 100.0
    # Python 3's built-in round() uses banker's rounding (round-half-to-even).
    # We document the policy explicitly because legal-style miss-risk numbers
    # must be reproducible across reviewers.
    return int(round(miss))


def format_no_evidence_reason(gold_recall: float) -> str:
    """Human-readable reason string for ``NO_EVIDENCE_IN_SEARCHED_CORPUS``.

    Example: ``"no clauses passed verification; recall=0.82, miss risk ~18%
    based on gold-set recall"``.
    """
    if not 0.0 <= gold_recall <= 1.0:
        msg = f"gold_recall must be in [0.0, 1.0], got {gold_recall}"
        raise ValueError(msg)
    miss_pct = miss_pct_from_recall(gold_recall)
    return (
        "no clauses passed verification; "
        f"recall={gold_recall:.2f}, miss risk ~{miss_pct}% based on gold-set recall"
    )


def format_insufficient_coverage_reason() -> str:
    """Canonical reason string for ``INSUFFICIENT_COVERAGE``."""
    return "no gold-set recall measured for this indicator"


def requires_authoritative_source_check(coverage_state: CoverageState) -> bool:
    """Return True iff this state should trigger a 'widen the corpus' prompt.

    Only ``INSUFFICIENT_COVERAGE`` does: ``EVIDENCE_FOUND`` already has
    evidence, and ``NO_EVIDENCE_IN_SEARCHED_CORPUS`` carries a measured recall
    that bounds the miss risk.
    """
    return coverage_state == CoverageState.INSUFFICIENT_COVERAGE
