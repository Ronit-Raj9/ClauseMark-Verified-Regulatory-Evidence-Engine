"""`CoverageReasonerPort` implementation — the 3-state absence reasoner.

The class :class:`CoverageReasoner` is pure: no I/O, no model calls, no
global state. It composes the helpers in :mod:`rie_coverage.policy` into a
``CoverageRecord`` for the orchestrator to persist.

A bare ``0`` is structurally impossible: the implementation can only
construct a ``CoverageRecord`` in one of three :class:`CoverageState`
branches, and the LLM is never consulted.
"""

from __future__ import annotations

from collections.abc import Sequence

from rie_contracts import (
    Claim,
    CoverageReasonerPort,
    CoverageRecord,
    CoverageState,
)

from rie_coverage.policy import (
    classify,
    filter_qualifying_claims,
    format_insufficient_coverage_reason,
    format_no_evidence_reason,
)


class CoverageReasoner(CoverageReasonerPort):
    """Production implementation of the 3-state absence reasoner.

    The constructor takes no dependencies: this reasoner is intentionally a
    pure function dressed as a class so it can plug behind the port without
    any wiring at call time.
    """

    def evaluate(
        self,
        jurisdiction: str,
        indicator_id: str,
        verified_claims: Sequence[Claim],
        gold_recall: float | None,
    ) -> CoverageRecord:
        """Return the :class:`CoverageRecord` for this ``(jurisdiction, indicator)``.

        Branching is fully covered by :func:`rie_coverage.policy.classify`;
        this method only materialises the record (reason text, measured
        recall, surviving claim ids).
        """
        qualifying = filter_qualifying_claims(verified_claims, indicator_id, jurisdiction)
        state = classify(verified_claims, gold_recall, indicator_id, jurisdiction)

        if state == CoverageState.EVIDENCE_FOUND:
            return CoverageRecord(
                jurisdiction=jurisdiction,
                indicator_id=indicator_id,
                state=CoverageState.EVIDENCE_FOUND,
                # measured_recall is preserved when present — it remains
                # useful context for downstream review even though we found
                # evidence.
                measured_recall=gold_recall,
                reason=None,
                verified_claim_ids=[c.claim_id for c in qualifying],
            )

        if state == CoverageState.NO_EVIDENCE_IN_SEARCHED_CORPUS:
            # Reachable only when gold_recall is not None — classify() enforces it.
            assert gold_recall is not None
            return CoverageRecord(
                jurisdiction=jurisdiction,
                indicator_id=indicator_id,
                state=CoverageState.NO_EVIDENCE_IN_SEARCHED_CORPUS,
                measured_recall=gold_recall,
                reason=format_no_evidence_reason(gold_recall),
                verified_claim_ids=[],
            )

        # INSUFFICIENT_COVERAGE — the only honest answer when we have neither
        # verified evidence nor a measured recall to bound the miss risk.
        return CoverageRecord(
            jurisdiction=jurisdiction,
            indicator_id=indicator_id,
            state=CoverageState.INSUFFICIENT_COVERAGE,
            measured_recall=None,
            reason=format_insufficient_coverage_reason(),
            verified_claim_ids=[],
        )
