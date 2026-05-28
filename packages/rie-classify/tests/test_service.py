"""End-to-end test of the classification service with a fake LLM."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

import pytest
from rie_classify.fake_llm import DeterministicFakeLlm
from rie_classify.llm_client import LlmClient
from rie_classify.service import ClassificationService
from rie_contracts import (
    ClassifierPort,
    ClausePattern,
    Element,
    IndicatorConfig,
    Layer1Status,
    PillarConfig,
    SpanRole,
)

INDICATOR_64 = "6.4"
INDICATOR_61 = "6.1"


def _fake_for_64() -> DeterministicFakeLlm:
    """Fake whose heuristics steer the canonical PDPA clause to 6.4."""
    return DeterministicFakeLlm(
        keyword_to_indicator={
            ("shall not", "comparable"): INDICATOR_64,
            ("shall not",): INDICATOR_64,
        }
    )


def test_service_implements_classifier_port(
    element_store: Callable[[str], Element],
) -> None:
    svc = ClassificationService(_fake_for_64(), element_store)
    assert isinstance(svc, ClassifierPort)


def test_canonical_pdpa_classifies_as_64_conditional_regime(
    clause_element: Element,
    neighbourhood: list[Element],
    pillar: PillarConfig,
    indicator_choices: list[IndicatorConfig],
    element_store: Callable[[str], Element],
) -> None:
    svc = ClassificationService(_fake_for_64(), element_store)
    claim = svc.classify_clause(
        clause_element=clause_element,
        neighbourhood=neighbourhood,
        pillar=pillar,
        indicator_choices=indicator_choices,
        n_samples=3,
    )

    assert claim.indicator_id == INDICATOR_64
    assert claim.clause_pattern is ClausePattern.CONDITIONAL_REGIME
    assert claim.pillar_id == pillar.pillar_id
    assert claim.clause_id == clause_element.element_id
    assert claim.layer1_status is Layer1Status.PENDING_VERIFICATION
    assert claim.self_consistency_votes[INDICATOR_64] >= 2

    # §6.2 decomposition carried on the claim.
    assert claim.decomposition.subject.strip()
    assert claim.decomposition.constraint.strip()

    # Evidence materialised from the element store, not from the LLM string.
    assert claim.evidence_spans, "evidence_spans must not be empty"
    primary_spans = [s for s in claim.evidence_spans if s.role is SpanRole.PRIMARY]
    assert len(primary_spans) == 1
    primary = primary_spans[0]
    assert primary.element_id == clause_element.element_id
    assert primary.doc_id == clause_element.doc_id
    assert primary.char_start == clause_element.char_start
    assert primary.char_end == clause_element.char_end
    # The span_id derives deterministically from the stored Element offsets.
    assert (
        primary.span_id
        == f"{clause_element.doc_id}#{clause_element.char_start}-{clause_element.char_end}"
    )

    # Regime assembly: primary first, then unique neighbour ids preserved.
    assert claim.regime.primary_element_id == clause_element.element_id
    assert clause_element.element_id in claim.regime.member_element_ids
    assert neighbourhood[0].element_id in claim.regime.member_element_ids


def test_claim_id_is_stable_md5(
    clause_element: Element,
    neighbourhood: list[Element],
    pillar: PillarConfig,
    indicator_choices: list[IndicatorConfig],
    element_store: Callable[[str], Element],
) -> None:
    import hashlib

    svc = ClassificationService(
        _fake_for_64(),
        element_store,
        now=lambda: datetime(2026, 5, 24, tzinfo=UTC),
    )
    claim = svc.classify_clause(
        clause_element, neighbourhood, pillar, indicator_choices, n_samples=3
    )
    expected = hashlib.md5(
        f"{clause_element.doc_id}|{clause_element.element_id}|{INDICATOR_64}".encode(),
        usedforsecurity=False,
    ).hexdigest()
    assert claim.claim_id == expected


def test_jurisdiction_resolver_is_used(
    clause_element: Element,
    neighbourhood: list[Element],
    pillar: PillarConfig,
    indicator_choices: list[IndicatorConfig],
    element_store: Callable[[str], Element],
) -> None:
    svc = ClassificationService(
        _fake_for_64(),
        element_store,
        get_jurisdiction=lambda _doc_id: "SAMPLE",
    )
    claim = svc.classify_clause(
        clause_element, neighbourhood, pillar, indicator_choices, n_samples=3
    )
    assert claim.jurisdiction == "SAMPLE"


def test_n_samples_validated() -> None:
    svc = ClassificationService(DeterministicFakeLlm(), lambda _e: None)  # type: ignore[arg-type, return-value]
    with pytest.raises(ValueError, match=">= 1"):
        svc.classify_clause(
            clause_element=_dummy_element(),
            neighbourhood=[],
            pillar=_dummy_pillar(),
            indicator_choices=_dummy_choices(),
            n_samples=0,
        )


def test_empty_indicator_choices_rejected(
    clause_element: Element,
    pillar: PillarConfig,
    element_store: Callable[[str], Element],
) -> None:
    svc = ClassificationService(DeterministicFakeLlm(), element_store)
    with pytest.raises(ValueError, match="non-empty"):
        svc.classify_clause(
            clause_element=clause_element,
            neighbourhood=[],
            pillar=pillar,
            indicator_choices=[],
            n_samples=1,
        )


def test_llm_failure_propagates_when_all_samples_fail(
    clause_element: Element,
    neighbourhood: list[Element],
    pillar: PillarConfig,
    indicator_choices: list[IndicatorConfig],
    element_store: Callable[[str], Element],
) -> None:
    class _AlwaysFailLlm(LlmClient):
        def complete_json(
            self,
            prompt: str,
            schema: dict[str, object],
            temperature: float,
            seed: int | None,
        ) -> dict[str, object]:
            raise RuntimeError("backend down")

    svc = ClassificationService(_AlwaysFailLlm(), element_store)
    with pytest.raises(RuntimeError, match="zero valid samples"):
        svc.classify_clause(clause_element, neighbourhood, pillar, indicator_choices, n_samples=3)


# ── helpers ─────────────────────────────────────────────────────────────────


def _dummy_element() -> Element:
    from rie_contracts import ElementType

    return Element(
        element_id="x",
        doc_id="d",
        element_type=ElementType.PARAGRAPH,
        text="t",
        page=1,
        char_start=0,
        char_end=1,
        extraction_confidence=1.0,
    )


def _dummy_pillar() -> PillarConfig:
    from rie_contracts import DocumentProfile

    return PillarConfig(
        pillar_id="p",
        pillar_name="p",
        cluster="c",
        document_profile=DocumentProfile.STATUTORY_LEGAL_TEXT,
        status="built",
        indicators=_dummy_choices(),
    )


def _dummy_choices() -> list[IndicatorConfig]:
    from rie_contracts import EvaluationMethod

    return [
        IndicatorConfig(
            indicator_id="z.1",
            name="z",
            definition="z",
            evaluation_method=EvaluationMethod.CLAUSE_EXTRACTION,
        )
    ]
