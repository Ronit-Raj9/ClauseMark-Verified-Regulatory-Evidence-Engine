"""KG entity-grounding — Phase 2 ADVISORY gate.

This module implements the §6.5 roadmap entity-grounding check as an
**advisory** verifier signal. Read this before relying on it:

Honest scope (§6.5)
-------------------
*"The KG / entity-grounding gate is roadmap, not MVP. A reliable legal KG
needs legal-tuned NER; stock spaCy will add noise."*

Accordingly this gate is **never** one of the required-4. Its
``GateResult`` (``gate=GateName.ENTITY_GROUNDING``) lands in
``VerificationReport.advisory_checks`` — it NEVER touches
``VerificationReport.status``. The required-4 gates (span-existence,
verbatim-match, entailment, self-consistency) remain the sole status
determinant. ``passed`` here merely *informs* a human reviewer; a failure
does not reject or flag the claim.

The concrete ``SpacyEntityExtractor`` lazy-loads ``spacy`` and the
``en_core_web_sm`` model. Stock spaCy is general-domain and will both miss
legal entities (statutory defined terms, cross-references) and emit
false-positive ``PERSON`` / ``ORG`` spans on legal prose — hence advisory.
``FakeEntityExtractor`` is provided for deterministic tests so the suite
never requires a spaCy install.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final, Protocol, runtime_checkable

from rie_contracts import Claim, GateName, GateResult
from rie_contracts.ports import ElementTextResolver

_DEFAULT_GROUNDING_THRESHOLD: Final[float] = 0.5

# spaCy entity labels worth grounding for legal text. NORP/LANGUAGE/etc. add
# pure noise; we keep the substantive ones. Advisory only — see module docstring.
_RELEVANT_SPACY_LABELS: Final[frozenset[str]] = frozenset(
    {"PERSON", "ORG", "GPE", "LAW", "NORP", "FAC", "PRODUCT", "EVENT", "DATE", "MONEY"}
)


@dataclass(frozen=True)
class Entity:
    """A surface entity: ``(text, label, char_start, char_end)``.

    Offsets are relative to the text the extractor was run over.
    """

    text: str
    label: str
    char_start: int
    char_end: int


@runtime_checkable
class EntityExtractor(Protocol):
    """Pluggable NER. Returns surface entities found in ``text``."""

    def extract(self, text: str) -> list[Entity]: ...


class SpacyEntityExtractor:
    """spaCy-backed extractor. Lazy-loads ``spacy`` + ``en_core_web_sm``.

    Raises a clear ``ImportError`` (caught by the checker → advisory skip)
    when spaCy or the model is unavailable. Stock spaCy is general-domain
    and noisy on legal text — this is documented per §6.5 and the result is
    advisory only.
    """

    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        self._model_name = model_name
        self._nlp: object | None = None

    def _ensure_loaded(self) -> object:
        if self._nlp is not None:
            return self._nlp
        try:
            import spacy  # noqa: PLC0415 — lazy, optional dependency
        except ImportError as exc:  # pragma: no cover - exercised only without spaCy
            msg = (
                "SpacyEntityExtractor requires the optional 'spacy' dependency. "
                "Install it with: uv add --package rie-verify spacy"
            )
            raise ImportError(msg) from exc
        try:
            self._nlp = spacy.load(self._model_name)
        except OSError as exc:  # pragma: no cover - model-not-downloaded path
            msg = (
                f"spaCy model '{self._model_name}' is not installed. "
                f"Download it with: python -m spacy download {self._model_name}"
            )
            raise ImportError(msg) from exc
        return self._nlp

    def extract(self, text: str) -> list[Entity]:
        nlp = self._ensure_loaded()
        doc = nlp(text)  # type: ignore[operator]
        out: list[Entity] = []
        for ent in doc.ents:  # type: ignore[attr-defined]
            if ent.label_ not in _RELEVANT_SPACY_LABELS:
                continue
            out.append(
                Entity(
                    text=ent.text,
                    label=ent.label_,
                    char_start=ent.start_char,
                    char_end=ent.end_char,
                )
            )
        return out


class FakeEntityExtractor:
    """Deterministic extractor for tests — returns canned entities verbatim."""

    def __init__(self, canned_entities: list[Entity]) -> None:
        self._canned = list(canned_entities)

    def extract(self, text: str) -> list[Entity]:
        del text  # canned extractor ignores input
        return list(self._canned)


def _normalise(text: str) -> str:
    return " ".join(text.lower().split())


class EntityGroundingChecker:
    """Advisory entity-grounding check.

    Extracts entities from the cited span text, then grounds each
    decomposition-mentioned entity against (a) the claim decomposition
    surface text and (b) the regime member element texts. ``score`` is the
    fraction of decomposition entities found grounded. ``passed`` is
    ``score >= threshold`` — but this is ADVISORY: ``passed`` only informs a
    reviewer and NEVER changes ``VerificationReport.status``.
    """

    def __init__(
        self,
        extractor: EntityExtractor,
        *,
        threshold: float = _DEFAULT_GROUNDING_THRESHOLD,
    ) -> None:
        self._extractor = extractor
        self._threshold = threshold

    @property
    def threshold(self) -> float:
        return self._threshold

    def check(
        self,
        claim: Claim,
        get_element_text: ElementTextResolver,
    ) -> GateResult:
        now = datetime.now(UTC)

        # spaCy / model absence → advisory skip, recorded as not-passed with a
        # clear reason. Never raises into the caller.
        try:
            grounding_blob = self._collect_grounding_text(claim, get_element_text)
            decomposition_text = self._decomposition_text(claim)
            decomposition_entities = self._extractor.extract(decomposition_text)
        except ImportError as exc:
            return GateResult(
                gate=GateName.ENTITY_GROUNDING,
                passed=False,
                detail=f"entity-grounding skipped (advisory): {exc}",
                ran_at=now,
            )

        if not decomposition_entities:
            # Nothing named to ground → vacuously grounded. Advisory pass.
            return GateResult(
                gate=GateName.ENTITY_GROUNDING,
                passed=True,
                detail="grounded 0/0 entities; advisory only (no named entities in decomposition)",
                score=1.0,
                ran_at=now,
            )

        normalised_blob = _normalise(grounding_blob)
        grounded = 0
        for ent in decomposition_entities:
            key = _normalise(ent.text)
            if key and key in normalised_blob:
                grounded += 1

        total = len(decomposition_entities)
        score = grounded / total
        passed = score >= self._threshold

        return GateResult(
            gate=GateName.ENTITY_GROUNDING,
            passed=passed,
            detail=f"grounded {grounded}/{total} entities; advisory only",
            score=score,
            ran_at=now,
        )

    # ──────────────────────────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────────────────────────

    def _collect_grounding_text(
        self,
        claim: Claim,
        get_element_text: ElementTextResolver,
    ) -> str:
        """Cited span text + regime member element texts, concatenated."""
        element_ids: list[str] = []
        seen: set[str] = set()
        for span in claim.evidence_spans:
            if span.element_id not in seen:
                seen.add(span.element_id)
                element_ids.append(span.element_id)
        for eid in claim.regime.member_element_ids:
            if eid not in seen:
                seen.add(eid)
                element_ids.append(eid)

        texts: list[str] = []
        for eid in element_ids:
            try:
                texts.append(get_element_text(eid))
            except (KeyError, LookupError, ValueError):
                continue
        return " ".join(texts)

    @staticmethod
    def _decomposition_text(claim: Claim) -> str:
        d = claim.decomposition
        parts: list[str] = [d.subject, d.constraint]
        if d.condition:
            parts.append(d.condition)
        if d.context:
            parts.append(d.context)
        return " . ".join(parts)


__all__ = [
    "Entity",
    "EntityExtractor",
    "EntityGroundingChecker",
    "FakeEntityExtractor",
    "SpacyEntityExtractor",
]
