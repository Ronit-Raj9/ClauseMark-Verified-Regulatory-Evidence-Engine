"""Constrained-decoding classification service — implements `ClassifierPort`.

Pipeline per call:
1. Build a structured prompt (clause + neighbourhood + indicator definitions).
2. Build a dynamic Pydantic model whose `indicator_id` is a `Literal` of the
   exact candidate ids — its JSON schema is what we hand to the grammar
   decoder so the model is physically unable to emit any other value.
3. Run the LLM ``n_samples`` times with varying temperature/seed for
   self-consistency. Each raw dict is validated against the Pydantic model;
   anything that fails validation is dropped (with the failure counted as a
   vote-less sample).
4. Tally votes via `Counter` over `indicator_id`. The modal classification
   wins; its structured fields (pattern, decomposition, evidence/regime ids)
   are the ones we use to materialise the Claim.
5. Materialise `EvidenceSpan`s from element_ids using the injected
   ``get_element`` resolver — the LLM never wrote a citation string.
6. Materialise `LegalRegime` from emitted ``regime_member_ids``.
7. Stable `claim_id` = md5(jurisdiction + element_id + indicator_id).hex.
8. `layer1_status = PENDING_VERIFICATION`; record `self_consistency_votes`.
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from rie_contracts import (
    Claim,
    ClassifierPort,
    ClausePattern,
    Element,
    EvidenceSpan,
    IndicatorConfig,
    Layer1Status,
    LegalRegime,
    PillarConfig,
    SpanRole,
)

from .config import ClassifierConfig
from .language import DEFAULT_LANGUAGE, detect_language
from .llm_client import LlmClient
from .prompt import build_classification_prompt
from .schema import ClassificationOutputBase, build_output_model


class InternalClassification(BaseModel):
    """Adapter-internal carrier that pairs the modal LLM output with the detected language.

    Not exposed via the ``ClassifierPort`` contract (which is frozen). Used
    only within the service to keep the language attached to the modal
    sample without smuggling it into the LLM JSON schema (which would let
    the LLM author a language tag — exactly what we forbid).
    """

    model_config = ConfigDict(frozen=True)

    language: str = Field(min_length=2, max_length=8)
    language_pillar_match: bool

logger = logging.getLogger(__name__)

# Sampling plan for self-consistency. Index 0 is deterministic; the rest add
# entropy via temperature *and* seed so we exercise both knobs.
_SAMPLING_PLAN: tuple[tuple[float, int], ...] = (
    (0.0, 0),
    (0.6, 1),
    (0.7, 2),
    (0.8, 3),
    (0.9, 4),
)


class ClassificationService(ClassifierPort):
    """Production-grade classifier — constrained decoding + self-consistency."""

    def __init__(
        self,
        llm: LlmClient,
        get_element: Callable[[str], Element],
        get_jurisdiction: Callable[[str], str] | None = None,
        now: Callable[[], datetime] | None = None,
        config: ClassifierConfig | None = None,
    ) -> None:
        self._llm = llm
        self._get_element = get_element
        # ClassifierPort.classify_clause does not take jurisdiction explicitly,
        # so the orchestrator wires a resolver (doc_id -> jurisdiction). The
        # default returns the doc_id, which keeps tests self-contained while
        # remaining deterministic.
        self._get_jurisdiction = get_jurisdiction or (lambda doc_id: doc_id)
        self._now = now or (lambda: datetime.now(UTC))
        self._config = config or ClassifierConfig()

    # ----------------------------------------------------------------- port
    def classify_clause(
        self,
        clause_element: Element,
        neighbourhood: Sequence[Element],
        pillar: PillarConfig,
        indicator_choices: Sequence[IndicatorConfig],
        n_samples: int = 3,
    ) -> Claim:
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")
        if not indicator_choices:
            raise ValueError("indicator_choices must be non-empty")

        detected_language = self._detect_clause_language(clause_element.text)
        _pillar_supports_language(
            indicator_choices=indicator_choices, language=detected_language
        )
        prompt = build_classification_prompt(
            clause_element=clause_element,
            neighbourhood=neighbourhood,
            pillar=pillar,
            indicator_choices=indicator_choices,
            detected_language=detected_language,
        )
        output_model = build_output_model(list(indicator_choices))
        schema = output_model.model_json_schema()

        samples: list[ClassificationOutputBase] = self._draw_samples(
            prompt=prompt, schema=schema, output_model=output_model, n_samples=n_samples
        )
        if not samples:
            raise RuntimeError(
                "Classification produced zero valid samples — every LLM call "
                "failed schema validation. This indicates either a broken "
                "LLM client or a misconfigured indicator enum."
            )

        votes = Counter(s.indicator_id for s in samples)
        winner_id, _ = votes.most_common(1)[0]
        modal = next(s for s in samples if s.indicator_id == winner_id)

        return self._build_claim(
            clause_element=clause_element,
            pillar=pillar,
            chosen=modal,
            votes=dict(votes),
        )

    # ----------------------------------------------------------------- impl
    def _detect_clause_language(self, text: str) -> str:
        if not self._config.multilingual_enabled:
            return DEFAULT_LANGUAGE
        if os.getenv("RIE_MULTILINGUAL_ENABLED", "1") == "0":
            return DEFAULT_LANGUAGE
        return detect_language(text)

    def _draw_samples(
        self,
        *,
        prompt: str,
        schema: dict[str, Any],
        output_model: type[ClassificationOutputBase],
        n_samples: int,
    ) -> list[ClassificationOutputBase]:
        samples: list[ClassificationOutputBase] = []
        for i in range(n_samples):
            temperature, seed = self._sampling_params(i)
            try:
                raw = self._llm.complete_json(
                    prompt=prompt, schema=schema, temperature=temperature, seed=seed
                )
            except Exception as exc:
                logger.warning("Classification sample %d failed at the LLM call: %s", i, exc)
                continue
            try:
                samples.append(output_model.model_validate(raw))
            except ValidationError as exc:
                # Real grammar-constrained backends should make this
                # impossible. If we see it, the backend ignored the schema.
                logger.warning(
                    "Classification sample %d failed Pydantic validation: %s",
                    i,
                    exc,
                )
                continue
        return samples

    @staticmethod
    def _sampling_params(i: int) -> tuple[float, int]:
        if i < len(_SAMPLING_PLAN):
            return _SAMPLING_PLAN[i]
        # Past the canned plan: temperature 0.8, seed = i (still deterministic).
        return (0.8, i)

    def _build_claim(
        self,
        *,
        clause_element: Element,
        pillar: PillarConfig,
        chosen: ClassificationOutputBase,
        votes: dict[str, int],
    ) -> Claim:
        evidence_spans = self._materialise_spans(
            element_ids=chosen.evidence_element_ids,
            primary_element_id=clause_element.element_id,
        )
        regime = self._materialise_regime(
            primary_element_id=clause_element.element_id,
            regime_member_ids=chosen.regime_member_ids,
        )

        jurisdiction = self._get_jurisdiction(clause_element.doc_id)
        claim_id = self._claim_id(
            jurisdiction=jurisdiction,
            element_id=clause_element.element_id,
            indicator_id=chosen.indicator_id,
        )

        return Claim(
            claim_id=claim_id,
            indicator_id=chosen.indicator_id,
            pillar_id=pillar.pillar_id,
            clause_id=clause_element.element_id,
            jurisdiction=jurisdiction,
            clause_pattern=ClausePattern(chosen.clause_pattern),
            decomposition=chosen.decomposition,
            evidence_spans=evidence_spans,
            regime=regime,
            layer1_status=Layer1Status.PENDING_VERIFICATION,
            self_consistency_votes=votes,
            created_at=self._now(),
        )

    def _materialise_spans(
        self,
        *,
        element_ids: Sequence[str],
        primary_element_id: str,
    ) -> list[EvidenceSpan]:
        spans: list[EvidenceSpan] = []
        for eid in element_ids:
            try:
                el = self._get_element(eid)
            except KeyError:
                logger.warning("Evidence element_id %s not in element store", eid)
                continue
            role = (
                SpanRole.PRIMARY if el.element_id == primary_element_id else SpanRole.REGIME_MEMBER
            )
            spans.append(
                EvidenceSpan(
                    span_id=f"{el.doc_id}#{el.char_start}-{el.char_end}",
                    element_id=el.element_id,
                    doc_id=el.doc_id,
                    char_start=el.char_start,
                    char_end=el.char_end,
                    role=role,
                )
            )
        if not spans:
            raise RuntimeError(
                "No evidence spans could be materialised — every emitted "
                "element_id was unknown to the element store."
            )
        return spans

    def _materialise_regime(
        self,
        *,
        primary_element_id: str,
        regime_member_ids: Sequence[str],
    ) -> LegalRegime:
        members: list[str] = [primary_element_id]
        for eid in regime_member_ids:
            if eid == primary_element_id:
                continue
            try:
                self._get_element(eid)  # validates existence
            except KeyError:
                logger.warning("Regime element_id %s not in element store", eid)
                continue
            members.append(eid)
        return LegalRegime(
            primary_element_id=primary_element_id,
            member_element_ids=members,
        )

    @staticmethod
    def _claim_id(*, jurisdiction: str, element_id: str, indicator_id: str) -> str:
        key = f"{jurisdiction}|{element_id}|{indicator_id}".encode()
        return hashlib.md5(key, usedforsecurity=False).hexdigest()


def _pillar_supports_language(
    *,
    indicator_choices: Sequence[IndicatorConfig],
    language: str,
) -> bool:
    """Return True when any indicator exposes keyword lists for ``language``."""
    for ind in indicator_choices:
        if language in ind.positive_keywords or language in ind.negative_cues:
            return True
    return False


__all__ = ["ClassificationService"]
