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

from rie_coverage.authoritative_sources import is_defensible_zero
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

    def evaluate_defensible(
        self,
        jurisdiction: str,
        indicator_id: str,
        verified_claims: Sequence[Claim],
        gold_recall: float | None,
        *,
        reachability: float | None = None,
    ) -> CoverageRecord:
        """Phase 2 §7 upgrade: same 3-state output, defensibility-aware reason.

        This is a strict superset of :meth:`evaluate` — the returned
        ``CoverageRecord`` schema is UNCHANGED (still one of exactly three
        :class:`CoverageState` values) and a bare ``0`` is STILL never emitted.
        The only difference appears in the ``NO_EVIDENCE_IN_SEARCHED_CORPUS``
        branch: the ``reason`` string is enriched with a defensibility verdict
        derived from :func:`is_defensible_zero` over the measured gold-set
        recall AND the authoritative-source ``reachability``.

        * When recall and reachability both clear their floors, the reason
          marks the result a **defensible-0 candidate** — a signal that a human
          reviewer MAY treat the absence as a real ``0``. The state is NOT
          changed and no ``0`` is written.
        * Otherwise the reason marks the result **bounded** "no evidence in
          searched corpus", noting the recall/reachability shortfall.

        ``reachability`` defaults to ``0.0`` when omitted, which can never on
        its own make an absence defensible — the conservative default.
        """
        base = self.evaluate(jurisdiction, indicator_id, verified_claims, gold_recall)

        if base.state != CoverageState.NO_EVIDENCE_IN_SEARCHED_CORPUS:
            return base

        # Reachable only when gold_recall is not None — evaluate() enforces it.
        assert gold_recall is not None
        reach = 0.0 if reachability is None else reachability
        defensible = is_defensible_zero(base.state, reach, gold_recall)

        if defensible:
            verdict = (
                f"defensible-0 candidate: gold-set recall {gold_recall:.2f} and "
                f"authoritative-source reachability {reach:.2f} both clear their "
                "floors, so a reviewer MAY treat this absence as a real 0 "
                "(still not auto-emitted)"
            )
        else:
            verdict = (
                f"bounded no-evidence: gold-set recall {gold_recall:.2f}, "
                f"authoritative-source reachability {reach:.2f} below floor — "
                "sources may be unreachable; NOT a defensible 0"
            )

        enriched_reason = f"{base.reason}; {verdict}"
        return base.model_copy(update={"reason": enriched_reason})
