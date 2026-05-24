"""The 4-gate verifier service. Implements `rie_contracts.VerifierPort`.

Honest semantics — see §6.5 / §6.7 of `systemArchitecture.md`:

* Tier A — deterministic, runs on every claim:
    1. span_existence — re-extract every cited span by offset; require non-empty.
    2. verbatim_match — re-extracted slice must be byte-identical to itself
       at the offsets stored in the span (a tautology over a single source of
       truth — that's the point: the LLM never authors the citation string,
       and this gate keeps it honest by re-doing the extraction).
* Tier B — model-based, runs only on Tier-A survivors:
    3. entailment — NLI + second LLM. **Any disagreement → flagged.**
       Pass iff (NLI entailment_prob > 0.6) AND (second_llm verdict is True).
    4. self_consistency — pre-computed N=3 votes carried on the Claim. Pass
       iff the winning vote crosses a strict-majority threshold AND the
       winner equals `claim.indicator_id`.

Status derivation (§6.7):
* deterministic gate fail (gates 1 or 2)            → REJECTED
* model gate fail (gates 3 or 4, all-deterministic-passed) → FLAGGED
* all four pass                                     → VERIFIED

### Offset convention (read this carefully)

`ElementTextResolver.__call__(element_id)` returns *only* the element text.
The verifier interprets `EvidenceSpan.char_start` / `char_end` as direct
indices into that string. The orchestrator is therefore responsible for
ensuring the resolver returns a string whose slice
`text[span.char_start:span.char_end]` reproduces the cited substring
(typically: store spans element-relative and return the element text). If
`span.char_end > len(text)` or `span.char_end < span.char_start`, the
span_existence gate fails — that mismatch is exactly what the gate is meant
to catch.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from typing import Final

from rie_contracts import (
    Claim,
    EvidenceSpan,
    GateName,
    GateResult,
    VerificationReport,
    VerificationStatus,
)
from rie_contracts.ports import ElementTextResolver

from rie_verify.kg_grounding import KgGroundingResult, run_kg_grounding_gate
from rie_verify.nli import NliBackend
from rie_verify.second_llm import SecondLlmBackend

# Tier-B thresholds — held here, not in pillar configs (these are engine-wide
# verifier hyper-parameters, not pillar-specific logic).
_NLI_ENTAILMENT_THRESHOLD: Final[float] = 0.6


class VerificationService:
    """Concrete `VerifierPort`. Constructor-injected NLI + second-LLM backends.

    Phase 2: an optional 5th gate ("kg_grounding") is wired in behind the
    ``kg_gate_enabled`` flag (default False — Phase 2 opt-in). It does NOT
    change the canonical 4-gate ``verify(...)`` behaviour or status derivation.
    To consume it, callers use the additive ``verify_with_kg(...)`` method,
    which returns the standard ``VerificationReport`` plus a side-channel
    ``KgGroundingResult``. The frozen ``rie_contracts.VerificationReport``
    rejects extra gate entries, so we deliberately do not append.
    """

    def __init__(
        self,
        nli_model: NliBackend,
        second_llm: SecondLlmBackend,
        *,
        nli_entailment_threshold: float = _NLI_ENTAILMENT_THRESHOLD,
        kg_gate_enabled: bool = False,
    ) -> None:
        self._nli: NliBackend = nli_model
        self._second_llm: SecondLlmBackend = second_llm
        self._nli_threshold: float = nli_entailment_threshold
        self._kg_gate_enabled: bool = kg_gate_enabled

    @property
    def kg_gate_enabled(self) -> bool:
        return self._kg_gate_enabled

    # ────────────────────────────────────────────────────────────────────
    # VerifierPort surface
    # ────────────────────────────────────────────────────────────────────

    def run_gate(
        self,
        gate: GateName,
        claim: Claim,
        get_element_text: ElementTextResolver,
    ) -> GateResult:
        match gate:
            case GateName.SPAN_EXISTENCE:
                return self._gate_span_existence(claim, get_element_text)
            case GateName.VERBATIM_MATCH:
                return self._gate_verbatim_match(claim, get_element_text)
            case GateName.ENTAILMENT:
                return self._gate_entailment(claim, get_element_text)
            case GateName.SELF_CONSISTENCY:
                return self._gate_self_consistency(claim)

    def verify(
        self,
        claim: Claim,
        get_element_text: ElementTextResolver,
    ) -> VerificationReport:
        # Tier A first — short-circuit nothing (we still want the gate result
        # for the report), but use the outcome to derive status.
        span_existence = self._gate_span_existence(claim, get_element_text)
        verbatim_match = self._gate_verbatim_match(claim, get_element_text)
        entailment = self._gate_entailment(claim, get_element_text)
        self_consistency = self._gate_self_consistency(claim)

        gates: list[GateResult] = [
            span_existence,
            verbatim_match,
            entailment,
            self_consistency,
        ]
        failure_reasons: list[str] = [f"{g.gate.value}: {g.detail}" for g in gates if not g.passed]

        deterministic_failed = not (span_existence.passed and verbatim_match.passed)
        model_failed = not (entailment.passed and self_consistency.passed)

        if deterministic_failed:
            status = VerificationStatus.REJECTED
        elif model_failed:
            status = VerificationStatus.FLAGGED
        else:
            status = VerificationStatus.VERIFIED

        return VerificationReport(
            claim_id=claim.claim_id,
            gates=gates,
            status=status,
            failure_reasons=failure_reasons,
        )

    # ────────────────────────────────────────────────────────────────────
    # Phase-2 opt-in — KG / entity-grounding (5th gate, side-channel)
    # ────────────────────────────────────────────────────────────────────

    def verify_with_kg(
        self,
        claim: Claim,
        get_element_text: ElementTextResolver,
        *,
        parent_of: Mapping[str, str | None] | None = None,
        extra_element_ids: Iterable[str] = (),
    ) -> tuple[VerificationReport, KgGroundingResult | None]:
        """Run the canonical 4 gates AND (if enabled) the KG-grounding gate.

        Returns the frozen-contract ``VerificationReport`` *plus* a
        ``KgGroundingResult`` (None when ``kg_gate_enabled`` is False).

        Status semantics:
          * deterministic failure (gates 1 or 2) → REJECTED — KG is not
            consulted (broken offsets make entity extraction meaningless).
          * model-gate failure (gates 3 or 4)    → FLAGGED.
          * all 4 pass + KG flag                 → downgraded to FLAGGED.
          * all 4 pass + KG pass                 → VERIFIED.

        We deliberately do NOT push the KG result into ``report.gates``: the
        ``rie_contracts.VerificationReport`` model_validator requires exactly
        the 4 canonical gate names — adding a 5th would break the contract.
        """
        report = self.verify(claim, get_element_text)

        if not self._kg_gate_enabled:
            return report, None

        # Skip KG when deterministic gates already rejected — broken offsets
        # poison the entity extractor and the human review can't act on KG
        # output anyway.
        if report.status is VerificationStatus.REJECTED:
            return report, None

        kg_result = run_kg_grounding_gate(
            claim,
            get_element_text,
            parent_of=parent_of,
            extra_element_ids=extra_element_ids,
        )

        if not kg_result.passed and report.status is VerificationStatus.VERIFIED:
            # Downgrade VERIFIED→FLAGGED on a KG flag. Build a fresh
            # VerificationReport (the model is not frozen but we treat the
            # original as immutable for downstream consumers).
            new_reasons = [*report.failure_reasons, f"{kg_result.gate}: {kg_result.detail}"]
            report = VerificationReport(
                claim_id=report.claim_id,
                gates=report.gates,
                status=VerificationStatus.FLAGGED,
                failure_reasons=new_reasons,
            )

        return report, kg_result

    # ────────────────────────────────────────────────────────────────────
    # Tier A — deterministic gates
    # ────────────────────────────────────────────────────────────────────

    def _gate_span_existence(
        self,
        claim: Claim,
        get_element_text: ElementTextResolver,
    ) -> GateResult:
        now = datetime.now(UTC)
        spans = claim.evidence_spans
        if not spans:
            return GateResult(
                gate=GateName.SPAN_EXISTENCE,
                passed=False,
                detail="claim has no evidence_spans",
                ran_at=now,
            )

        problems: list[str] = []
        for span in spans:
            try:
                text = get_element_text(span.element_id)
            except (KeyError, LookupError, ValueError) as exc:
                problems.append(f"{span.span_id}: element not resolvable ({exc})")
                continue

            if span.char_end < span.char_start:
                problems.append(f"{span.span_id}: char_end < char_start")
                continue
            if span.char_end > len(text):
                problems.append(
                    f"{span.span_id}: char_end {span.char_end} exceeds element "
                    f"text length {len(text)}"
                )
                continue
            slice_ = text[span.char_start : span.char_end]
            if not slice_:
                problems.append(f"{span.span_id}: resolved slice is empty")

        if problems:
            return GateResult(
                gate=GateName.SPAN_EXISTENCE,
                passed=False,
                detail="; ".join(problems),
                ran_at=now,
            )
        return GateResult(
            gate=GateName.SPAN_EXISTENCE,
            passed=True,
            detail=f"all {len(spans)} spans resolved",
            ran_at=now,
        )

    def _gate_verbatim_match(
        self,
        claim: Claim,
        get_element_text: ElementTextResolver,
    ) -> GateResult:
        """Re-extract every span by offset; require it equals itself.

        Per §6.6, the claim never carries an LLM-authored snippet — citations
        are materialised downstream by ID replacement. This gate proves the
        re-extraction is stable: we slice the element text twice (once to
        produce the expected snippet, once to compare) and require byte
        equality. A divergence here means the element text changed between
        calls, which would be a data-integrity bug.
        """
        now = datetime.now(UTC)
        spans = claim.evidence_spans
        if not spans:
            return GateResult(
                gate=GateName.VERBATIM_MATCH,
                passed=False,
                detail="claim has no evidence_spans",
                ran_at=now,
            )

        problems: list[str] = []
        for span in spans:
            try:
                text_a = get_element_text(span.element_id)
                text_b = get_element_text(span.element_id)
            except (KeyError, LookupError, ValueError) as exc:
                problems.append(f"{span.span_id}: element not resolvable ({exc})")
                continue

            if (
                span.char_end > len(text_a)
                or span.char_end > len(text_b)
                or span.char_end < span.char_start
            ):
                problems.append(f"{span.span_id}: offsets out of range")
                continue

            snippet_a = text_a[span.char_start : span.char_end]
            snippet_b = text_b[span.char_start : span.char_end]
            if snippet_a != snippet_b:
                problems.append(f"{span.span_id}: re-extraction not byte-identical")

        if problems:
            return GateResult(
                gate=GateName.VERBATIM_MATCH,
                passed=False,
                detail="; ".join(problems),
                ran_at=now,
            )
        return GateResult(
            gate=GateName.VERBATIM_MATCH,
            passed=True,
            detail=f"all {len(spans)} spans byte-identical on re-extraction",
            ran_at=now,
        )

    # ────────────────────────────────────────────────────────────────────
    # Tier B — model gates
    # ────────────────────────────────────────────────────────────────────

    def _gate_entailment(
        self,
        claim: Claim,
        get_element_text: ElementTextResolver,
    ) -> GateResult:
        """Run NLI + second-LLM; **any disagreement → flagged**.

        Pass iff:
          * NLI entailment_prob > threshold (default 0.6), AND
          * second_llm.judge_entailment returns True.

        Any other combination — both fail, or *either* fails — is a fail with
        an explicit "NLI/LLM disagree" detail when they disagree, so the
        reviewer sees exactly why the claim was flagged.
        """
        now = datetime.now(UTC)
        primary_span = self._primary_span(claim)
        if primary_span is None:
            return GateResult(
                gate=GateName.ENTAILMENT,
                passed=False,
                detail="claim has no evidence_spans",
                ran_at=now,
            )

        try:
            premise = get_element_text(primary_span.element_id)
        except (KeyError, LookupError, ValueError) as exc:
            return GateResult(
                gate=GateName.ENTAILMENT,
                passed=False,
                detail=f"primary span element not resolvable ({exc})",
                ran_at=now,
            )

        hypothesis = self._render_decomposition(claim)

        try:
            scores = self._nli.score(premise, hypothesis)
            llm_verdict = self._second_llm.judge_entailment(premise, hypothesis)
        except (RuntimeError, OSError, ValueError) as exc:
            return GateResult(
                gate=GateName.ENTAILMENT,
                passed=False,
                detail=f"model-gate backend failure: {exc}",
                ran_at=now,
            )

        nli_pass = scores["entailment"] > self._nli_threshold

        if nli_pass and llm_verdict:
            return GateResult(
                gate=GateName.ENTAILMENT,
                passed=True,
                detail=(
                    f"NLI entailment={scores['entailment']:.3f} > "
                    f"{self._nli_threshold} and second-LLM agrees"
                ),
                score=scores["entailment"],
                ran_at=now,
            )

        if nli_pass != llm_verdict:
            detail = (
                f"NLI/LLM disagree (NLI entailment={scores['entailment']:.3f}, "
                f"NLI pass={nli_pass}; second-LLM verdict={llm_verdict})"
            )
        else:
            detail = (
                f"both verifiers reject "
                f"(NLI entailment={scores['entailment']:.3f}, second-LLM verdict={llm_verdict})"
            )

        return GateResult(
            gate=GateName.ENTAILMENT,
            passed=False,
            detail=detail,
            score=scores["entailment"],
            ran_at=now,
        )

    def _gate_self_consistency(self, claim: Claim) -> GateResult:
        """N=3 sampling votes carried on the claim; require strict majority for the claimed indicator.

        Threshold rule — strict majority (`floor(N/2) + 1`) at every N:
          * N = 3 → require >= 2 votes for the winner.
          * N = 5 → require >= 3 votes for the winner.
          * N = 1 → require 1 vote (trivially passes if recorded).
        This collapses to `> N/2` and matches the §6.5 spec ("unstable label →
        flagged"): the moment the model can't repeat itself a majority of the
        time, the claim is unstable.

        Pass requires both (a) the threshold is met AND (b) the winning indicator
        equals `claim.indicator_id`.
        """
        now = datetime.now(UTC)
        votes = claim.self_consistency_votes
        if not votes:
            return GateResult(
                gate=GateName.SELF_CONSISTENCY,
                passed=False,
                detail="unstable: no self_consistency_votes recorded",
                ran_at=now,
            )

        total = sum(votes.values())
        if total <= 0:
            return GateResult(
                gate=GateName.SELF_CONSISTENCY,
                passed=False,
                detail="unstable: vote total is zero",
                ran_at=now,
            )

        winner, winner_count = Counter(votes).most_common(1)[0]
        required = (total // 2) + 1  # strict majority

        winner_disagrees = winner != claim.indicator_id
        threshold_missed = winner_count < required

        if threshold_missed and winner_disagrees:
            detail = (
                f"unstable: winner {winner} with {winner_count}/{total} votes below "
                f"threshold {required} AND winner != claim.indicator_id "
                f"{claim.indicator_id}"
            )
            return GateResult(
                gate=GateName.SELF_CONSISTENCY,
                passed=False,
                detail=detail,
                score=winner_count / total,
                ran_at=now,
            )

        if threshold_missed:
            return GateResult(
                gate=GateName.SELF_CONSISTENCY,
                passed=False,
                detail=(
                    f"unstable: winner {winner} with {winner_count}/{total} votes "
                    f"below threshold {required}"
                ),
                score=winner_count / total,
                ran_at=now,
            )

        if winner_disagrees:
            return GateResult(
                gate=GateName.SELF_CONSISTENCY,
                passed=False,
                detail=(
                    f"unstable: winning vote {winner} != claim.indicator_id {claim.indicator_id}"
                ),
                score=winner_count / total,
                ran_at=now,
            )

        return GateResult(
            gate=GateName.SELF_CONSISTENCY,
            passed=True,
            detail=f"stable: {winner_count}/{total} votes for {winner}",
            score=winner_count / total,
            ran_at=now,
        )

    # ────────────────────────────────────────────────────────────────────
    # Internals
    # ────────────────────────────────────────────────────────────────────

    @staticmethod
    def _primary_span(claim: Claim) -> EvidenceSpan | None:
        if not claim.evidence_spans:
            return None
        for span in claim.evidence_spans:
            if span.role.value == "primary":
                return span
        return claim.evidence_spans[0]

    @staticmethod
    def _render_decomposition(claim: Claim) -> str:
        """Render the decomposition as a single sentence for NLI / second-LLM consumption.

        Format mirrors Compliance-to-Code clause decomposition:
          "{subject} [when {condition}] must {constraint} [context: {context}]."
        """
        d = claim.decomposition
        parts: list[str] = [d.subject]
        if d.condition:
            parts.append(f"when {d.condition}")
        parts.append(f"must {d.constraint}")
        sentence = " ".join(parts).strip()
        if d.context:
            sentence = f"{sentence} (context: {d.context})"
        if not sentence.endswith("."):
            sentence = f"{sentence}."
        return sentence
