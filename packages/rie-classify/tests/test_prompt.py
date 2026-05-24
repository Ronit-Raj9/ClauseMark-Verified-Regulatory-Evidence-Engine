"""Tests for the pure prompt builder."""

from __future__ import annotations

from rie_classify.prompt import (
    SYSTEM_INSTRUCTION,
    build_classification_prompt,
)
from rie_contracts import Element, IndicatorConfig, PillarConfig


def test_prompt_contains_system_instruction(
    clause_element: Element,
    neighbourhood: list[Element],
    pillar: PillarConfig,
    indicator_choices: list[IndicatorConfig],
) -> None:
    prompt = build_classification_prompt(clause_element, neighbourhood, pillar, indicator_choices)
    assert SYSTEM_INSTRUCTION in prompt


def test_prompt_lists_every_indicator_id_and_definition(
    clause_element: Element,
    neighbourhood: list[Element],
    pillar: PillarConfig,
    indicator_choices: list[IndicatorConfig],
) -> None:
    prompt = build_classification_prompt(clause_element, neighbourhood, pillar, indicator_choices)
    for ic in indicator_choices:
        assert f"id: {ic.indicator_id}" in prompt
        # Definition text bleeds into the prompt.
        assert ic.definition.strip().split(".")[0] in prompt


def test_prompt_includes_allowed_ids_inline(
    clause_element: Element,
    neighbourhood: list[Element],
    pillar: PillarConfig,
    indicator_choices: list[IndicatorConfig],
) -> None:
    prompt = build_classification_prompt(clause_element, neighbourhood, pillar, indicator_choices)
    allowed = ", ".join(ic.indicator_id for ic in indicator_choices)
    assert f"the ONLY allowed indicator_id values: {allowed}" in prompt


def test_prompt_includes_clause_text_and_element_ids(
    clause_element: Element,
    neighbourhood: list[Element],
    pillar: PillarConfig,
    indicator_choices: list[IndicatorConfig],
) -> None:
    prompt = build_classification_prompt(clause_element, neighbourhood, pillar, indicator_choices)
    assert clause_element.text in prompt
    assert f"element_id: {clause_element.element_id}" in prompt
    for n in neighbourhood:
        assert f"element_id: {n.element_id}" in prompt
        assert n.text in prompt


def test_prompt_includes_pillar_metadata(
    clause_element: Element,
    neighbourhood: list[Element],
    pillar: PillarConfig,
    indicator_choices: list[IndicatorConfig],
) -> None:
    prompt = build_classification_prompt(clause_element, neighbourhood, pillar, indicator_choices)
    assert f"pillar_id: {pillar.pillar_id}" in prompt
    assert pillar.pillar_name in prompt
    assert pillar.cluster in prompt


def test_prompt_handles_empty_neighbourhood(
    clause_element: Element,
    pillar: PillarConfig,
    indicator_choices: list[IndicatorConfig],
) -> None:
    prompt = build_classification_prompt(clause_element, [], pillar, indicator_choices)
    assert "(no neighbourhood elements provided)" in prompt


def test_prompt_includes_few_shot_examples(
    clause_element: Element,
    neighbourhood: list[Element],
    pillar: PillarConfig,
    indicator_choices: list[IndicatorConfig],
) -> None:
    prompt = build_classification_prompt(clause_element, neighbourhood, pillar, indicator_choices)
    assert "few_shot_examples" in prompt
