"""Adversarial refutation panel — v4 citation guarantee (ADVISORY, can DEMOTE only).

Why this exists
---------------
The required-4 gates (span-existence, verbatim-match, entailment, self-consistency)
prove that a cited span *exists*, is *byte-identical*, and is *entailed* by the
clause's own decomposition. They do NOT prove the mapping ``clause -> indicator``
is the *right* one. ESCAP's own trials (Assignment-1 "why wrong" set, see
``rdtii-worked-examples`` / ``rdtii-pillar6-7-real-indicators``) failed on exactly
this class of error — **misinterpretation of law**:

  * A confidentiality duty (Banking Act s.47) mischaracterised as a
    data-localisation / transfer ban.
  * A cybersecurity *notice* cited as evidence of the *lack* of a cybersecurity
    framework (the cited measure proves the opposite of the claim).
  * A government-data measure scored against an indicator whose scope excludes
    government data.
  * A draft / repealed instrument cited as if in force.
  * A paraphrase passed off as a verbatim quote.

Each of those slips past the required-4 because the span really does exist and
really does entail the (wrongly framed) decomposition. The fix is an
**adversarial panel**: N skeptics, one per *lens* (a known failure mode), each of
which actively tries to REFUTE the mapping. A mapping that survives the panel is
more trustworthy; a mapping a majority of skeptics can refute is withheld.

Contract: DEMOTE-only — never PROMOTE
-------------------------------------
The panel is ADVISORY with respect to the required-4. Concretely:

  * Its ``GateResult`` (``gate=GateName.ENTITY_GROUNDING``, the only advisory
    slot the frozen contract allows) lands in
    ``VerificationReport.advisory_checks``.
  * It can **DEMOTE** a claim that the required-4 marked ``VERIFIED`` down to
    ``FLAGGED`` (withhold) when a majority of skeptics refute the mapping.
  * It can **NEVER PROMOTE**: a ``REJECTED`` claim (a deterministic-gate
    failure) stays ``REJECTED``; a ``FLAGGED`` claim stays ``FLAGGED``. The
    panel only ever moves status *toward* review, never away from it.

Default-to-refute on uncertainty: a skeptic that errors / is unsure counts as a
refutation. We would rather withhold a borderline mapping than auto-confirm it.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final, Protocol, runtime_checkable

import httpx
from rie_contracts import GateName, GateResult

# The lenses are the canonical failure-mode taxonomy (ESCAP Assignment-1 + judges'
# risk list). They are NOT pillar/indicator logic — they name *categories of
# misinterpretation*, so they are legitimately engine-side. No indicator id, no
# pillar id, no jurisdiction is hard-coded here.
DEFAULT_LENSES: Final[tuple[str, ...]] = (
    "wrong_indicator",
    "confidentiality_not_localisation",
    "government_data_scope_exception",
    "draft_or_repealed_not_in_force",
    "paraphrase_not_verbatim",
)


@runtime_checkable
class RefutationLlm(Protocol):
    """A skeptic. Given a candidate mapping and one lens, try to REFUTE it.

    Returns ``(refuted, reason)``. ``refuted=True`` means: under this lens, the
    mapping ``clause -> indicator`` looks WRONG. Implementations MUST
    default-to-refute on uncertainty / error (the panel relies on this).
    """

    def refute(
        self,
        claim_text: str,
        indicator_def: str,
        span_text: str,
        lens: str,
    ) -> tuple[bool, str]: ...


# Per-lens skeptic instructions. Engine-side text, no pillar/indicator literals.
_LENS_INSTRUCTIONS: Final[dict[str, str]] = {
    "wrong_indicator": (
        "Argue that the cited span maps to a DIFFERENT indicator than the one "
        "claimed. Refute if the span's subject-matter does not match the "
        "indicator definition's core legal question."
    ),
    "confidentiality_not_localisation": (
        "Argue the span is a confidentiality / secrecy duty being "
        "mischaracterised as a data-localisation or cross-border-transfer "
        "restriction. A duty not to DISCLOSE is not a requirement to STORE or "
        "PROCESS locally. Refute if the span is about secrecy, not location."
    ),
    "government_data_scope_exception": (
        "Argue the measure applies to GOVERNMENT / public-sector data and the "
        "indicator's scope excludes government data. Refute if the span is "
        "scoped to government data and the indicator does not score it."
    ),
    "draft_or_repealed_not_in_force": (
        "Argue the cited instrument is a draft, bill, consultation, repealed, or "
        "superseded text — not law in force. Refute if there is any sign the "
        "span is not from a currently-effective instrument."
    ),
    "paraphrase_not_verbatim": (
        "Argue the claim text paraphrases or overstates the span rather than "
        "quoting it. Refute if the claim asserts an obligation/prohibition the "
        "span does not literally state."
    ),
}


def _lens_instruction(lens: str) -> str:
    return _LENS_INSTRUCTIONS.get(
        lens,
        "Try to find any reason the cited span does not support the claimed mapping.",
    )


_REFUTE_PROMPT: Final[str] = """\
You are a skeptical legal-mapping reviewer. Your job is to REFUTE a proposed
mapping of a legal span to an indicator. Be adversarial: assume the mapping is
WRONG and look for evidence. If you are unsure, REFUTE.

LENS (the specific failure mode you are hunting for):
{lens_instruction}

INDICATOR DEFINITION:
{indicator_def}

CLAIM (what is asserted about the span):
{claim_text}

CITED SPAN (verbatim legal text):
{span_text}

Answer on the FIRST line with exactly "REFUTE" or "SURVIVE", then on the next
line a one-sentence reason."""


class OllamaRefuter:
    """Skeptic backed by a local Ollama server.

    Per §6.5 / the second-LLM rule, the refuter model MUST be a *different model
    family* than the primary classifier (and ideally than the entailment second
    LLM) so the panel adds independent signal rather than echoing the author.
    The model name + endpoint are env-configured via :meth:`from_env`.

    Network / parse failure → default-to-refute (uncertainty == refutation), so
    a flaky skeptic withholds rather than rubber-stamps.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout_s: float = 30.0,
    ) -> None:
        self._model: str = model
        self._base_url: str = base_url.rstrip("/")
        self._timeout_s: float = timeout_s

    @classmethod
    def from_env(cls) -> OllamaRefuter:
        """Build from ``RIE_REFUTER_MODEL`` / ``RIE_REFUTER_BASE_URL`` / ``RIE_REFUTER_TIMEOUT_S``.

        ``RIE_REFUTER_MODEL`` defaults to a different family than the classifier
        / entailment second LLM. The caller is responsible for ensuring family
        separation when overriding.
        """
        model = os.environ.get("RIE_REFUTER_MODEL", "mistral")
        base_url = os.environ.get("RIE_REFUTER_BASE_URL", "http://localhost:11434")
        timeout_s = float(os.environ.get("RIE_REFUTER_TIMEOUT_S", "30"))
        return cls(model=model, base_url=base_url, timeout_s=timeout_s)

    def refute(
        self,
        claim_text: str,
        indicator_def: str,
        span_text: str,
        lens: str,
    ) -> tuple[bool, str]:
        prompt = _REFUTE_PROMPT.format(
            lens_instruction=_lens_instruction(lens),
            indicator_def=indicator_def,
            claim_text=claim_text,
            span_text=span_text,
        )
        url = f"{self._base_url}/api/generate"
        payload: dict[str, object] = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 64},
        }
        try:
            with httpx.Client(timeout=self._timeout_s) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            # Default-to-refute on uncertainty: a skeptic that cannot speak
            # counts against the mapping.
            return True, f"refuter unreachable, defaulting to refute ({exc})"

        text = str(body.get("response", "")).strip()
        first_line = text.splitlines()[0].strip().lower() if text else ""
        reason = text.strip() or "no reason returned"
        if first_line.startswith("survive"):
            return False, reason
        # Anything that is not an explicit SURVIVE is a refutation (default-to-refute).
        return True, reason


class FakeRefuter:
    """Deterministic skeptic for tests — scripted verdicts keyed by lens.

    ``verdicts`` maps ``lens -> (refuted, reason)``. A lens absent from the map
    falls back to ``default`` (default-to-refute is the safe default, but the
    explicit default lets tests build all-survive panels easily).
    """

    def __init__(
        self,
        verdicts: dict[str, tuple[bool, str]] | None = None,
        *,
        default: tuple[bool, str] = (True, "no scripted verdict; defaulting to refute"),
    ) -> None:
        self._verdicts: dict[str, tuple[bool, str]] = dict(verdicts or {})
        self._default: tuple[bool, str] = default

    def refute(
        self,
        claim_text: str,
        indicator_def: str,
        span_text: str,
        lens: str,
    ) -> tuple[bool, str]:
        del claim_text, indicator_def, span_text  # scripted: input ignored
        return self._verdicts.get(lens, self._default)


@dataclass(frozen=True)
class LensVerdict:
    """One skeptic's verdict for one lens."""

    lens: str
    refuted: bool
    reason: str


@dataclass(frozen=True)
class PanelOutcome:
    """Structured panel result, retained alongside the advisory ``GateResult``."""

    survives: bool
    score: float
    verdicts: tuple[LensVerdict, ...]

    @property
    def refuting_lenses(self) -> tuple[str, ...]:
        return tuple(v.lens for v in self.verdicts if v.refuted)


class AdversarialPanel:
    """Runs N skeptics (one per lens) and aggregates by majority.

    A mapping **survives** iff *fewer than a majority* of lenses refute it (i.e.
    ``refute_count < majority``, where ``majority = floor(N/2) + 1``). ``score``
    is the fraction of lenses that did NOT refute. The advisory ``GateResult``
    uses ``gate=GateName.ENTITY_GROUNDING`` (the only advisory slot the frozen
    ``rie_contracts`` model permits) and lists which lenses refuted, with reasons.

    ADVISORY / DEMOTE-only: the panel result NEVER promotes a claim. The wiring
    in :class:`rie_verify.service.VerificationService` only ever uses a
    majority-refute to DEMOTE a ``VERIFIED`` claim to ``FLAGGED``.
    """

    def __init__(
        self,
        refuter: RefutationLlm,
        lenses: Sequence[str] = DEFAULT_LENSES,
    ) -> None:
        self._refuter: RefutationLlm = refuter
        self._lenses: tuple[str, ...] = tuple(lenses)
        if not self._lenses:
            msg = "AdversarialPanel requires at least one lens"
            raise ValueError(msg)

    @property
    def lenses(self) -> tuple[str, ...]:
        return self._lenses

    def evaluate(
        self,
        claim_text: str,
        indicator_def: str,
        span_text: str,
    ) -> PanelOutcome:
        """Run every skeptic and aggregate; pure (no GateResult), for callers
        that want the structured outcome."""
        verdicts: list[LensVerdict] = []
        for lens in self._lenses:
            try:
                refuted, reason = self._refuter.refute(claim_text, indicator_def, span_text, lens)
            except (RuntimeError, OSError, ValueError) as exc:
                # Default-to-refute on uncertainty.
                refuted, reason = True, f"skeptic raised, defaulting to refute ({exc})"
            verdicts.append(LensVerdict(lens=lens, refuted=bool(refuted), reason=reason))

        n = len(verdicts)
        refute_count = sum(1 for v in verdicts if v.refuted)
        majority = (n // 2) + 1  # strict majority
        survives = refute_count < majority
        score = (n - refute_count) / n
        return PanelOutcome(survives=survives, score=score, verdicts=tuple(verdicts))

    def run(
        self,
        claim_text: str,
        indicator_def: str,
        span_text: str,
    ) -> GateResult:
        """Run the panel and return the ADVISORY ``GateResult``.

        ``passed`` mirrors ``survives``. ``detail`` lists the refuting lenses and
        their reasons. The gate name is ``ENTITY_GROUNDING`` (the advisory slot);
        the service appends this to ``advisory_checks`` and uses a non-survival
        (majority-refute) ONLY to demote a VERIFIED claim to FLAGGED.
        """
        now = datetime.now(UTC)
        outcome = self.evaluate(claim_text, indicator_def, span_text)

        refute_count = sum(1 for v in outcome.verdicts if v.refuted)
        n = len(outcome.verdicts)
        if outcome.survives:
            detail = (
                f"adversarial panel: survived ({refute_count}/{n} lenses refuted; "
                f"advisory, demote-only)"
            )
        else:
            refuters = "; ".join(f"{v.lens}: {v.reason}" for v in outcome.verdicts if v.refuted)
            detail = (
                f"adversarial panel: majority-refuted ({refute_count}/{n}); "
                f"refuting lenses -> {refuters}"
            )

        return GateResult(
            gate=GateName.ENTITY_GROUNDING,
            passed=outcome.survives,
            detail=detail,
            score=outcome.score,
            ran_at=now,
        )


__all__ = [
    "DEFAULT_LENSES",
    "AdversarialPanel",
    "FakeRefuter",
    "LensVerdict",
    "OllamaRefuter",
    "PanelOutcome",
    "RefutationLlm",
]
