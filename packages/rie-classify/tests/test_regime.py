"""Tests for regime assembly helpers (§6.4)."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from rie_classify.fake_llm import DeterministicFakeLlm
from rie_classify.regime import (
    build_regime_assembly_input,
    materialise_regime_from_llm_ids,
    merge_llm_regime_hints,
)
from rie_classify.service import ClassificationService
from rie_contracts import Element, IndicatorConfig, LegalRegime, PillarConfig


def test_build_regime_assembly_input(
    clause_element: Element,
    definition_element: Element,
) -> None:
    payload = build_regime_assembly_input(
        clause_element,
        [clause_element.element_id, definition_element.element_id],
    )
    assert payload.primary_element_id == clause_element.element_id
    assert payload.doc_id == clause_element.doc_id
    assert definition_element.element_id in payload.llm_regime_member_ids


def test_materialise_regime_from_llm_ids(
    clause_element: Element,
    definition_element: Element,
    element_store: Callable[[str], Element],
) -> None:
    regime = materialise_regime_from_llm_ids(
        primary_element_id=clause_element.element_id,
        regime_member_ids=[definition_element.element_id],
        get_element=element_store,
    )
    assert regime.primary_element_id == clause_element.element_id
    assert definition_element.element_id in regime.member_element_ids
    assert definition_element.element_id in regime.definitions


def test_merge_llm_regime_hints_adds_missing_llm_members(
    clause_element: Element,
    definition_element: Element,
    element_store: Callable[[str], Element],
) -> None:
    graph_only = LegalRegime(
        primary_element_id=clause_element.element_id,
        member_element_ids=[clause_element.element_id],
    )
    merged = merge_llm_regime_hints(
        graph_only,
        [definition_element.element_id],
        get_element=element_store,
    )
    assert definition_element.element_id in merged.member_element_ids
    assert definition_element.element_id in merged.definitions


def test_regime_assembler_receives_llm_member_ids(
    clause_element: Element,
    neighbourhood: list[Element],
    pillar: PillarConfig,
    indicator_choices: list[IndicatorConfig],
    element_store: Callable[[str], Element],
) -> None:
    received: list[tuple[str, tuple[str, ...]]] = []

    def _assembler(clause: Element, llm_ids: Sequence[str]) -> LegalRegime:
        received.append((clause.element_id, tuple(llm_ids)))
        return LegalRegime(
            primary_element_id=clause.element_id,
            member_element_ids=[clause.element_id],
        )

    svc = ClassificationService(
        DeterministicFakeLlm(keyword_to_indicator={("shall not", "comparable"): "6.4"}),
        element_store,
        regime_assembler=_assembler,
    )
    claim = svc.classify_clause(
        clause_element,
        neighbourhood,
        pillar,
        indicator_choices,
        n_samples=3,
    )

    assert received, "regime_assembler must be invoked"
    primary, llm_ids = received[0]
    assert primary == clause_element.element_id
    assert neighbourhood[0].element_id in llm_ids
    assert neighbourhood[0].element_id in claim.regime.member_element_ids


def test_legacy_single_arg_regime_assembler_still_works(
    clause_element: Element,
    neighbourhood: list[Element],
    pillar: PillarConfig,
    indicator_choices: list[IndicatorConfig],
    element_store: Callable[[str], Element],
) -> None:
    def _legacy_assembler(clause: Element) -> LegalRegime:
        return LegalRegime(
            primary_element_id=clause.element_id,
            member_element_ids=[clause.element_id],
        )

    svc = ClassificationService(
        DeterministicFakeLlm(keyword_to_indicator={("shall not", "comparable"): "6.4"}),
        element_store,
        regime_assembler=_legacy_assembler,  # type: ignore[arg-type]
    )
    claim = svc.classify_clause(
        clause_element,
        neighbourhood,
        pillar,
        indicator_choices,
        n_samples=3,
    )
    assert claim.regime.primary_element_id == clause_element.element_id
    assert neighbourhood[0].element_id in claim.regime.member_element_ids
