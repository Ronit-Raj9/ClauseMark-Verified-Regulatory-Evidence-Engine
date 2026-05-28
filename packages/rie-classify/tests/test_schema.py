"""Tests for the dynamic `ClassificationOutput` Pydantic model."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from rie_classify.schema import ClassificationOutputBase, build_output_model
from rie_contracts import ClausePattern, EvaluationMethod, IndicatorConfig

INDICATOR_A = "6.4"
INDICATOR_B = "6.2"


def _valid_payload(indicator_id: str) -> dict[str, object]:
    return {
        "indicator_id": indicator_id,
        "clause_pattern": "conditional_regime",
        "decomposition": {
            "subject": "organisation",
            "condition": "outbound transfer",
            "constraint": "comparable protection",
        },
        "evidence_element_ids": ["el1"],
        "regime_member_ids": ["el1"],
    }


@pytest.fixture
def choices() -> list[IndicatorConfig]:
    return [
        IndicatorConfig(
            indicator_id=INDICATOR_A,
            name="A",
            definition="A.",
            evaluation_method=EvaluationMethod.CLAUSE_EXTRACTION,
        ),
        IndicatorConfig(
            indicator_id=INDICATOR_B,
            name="B",
            definition="B.",
            evaluation_method=EvaluationMethod.CLAUSE_EXTRACTION,
        ),
    ]


def test_build_output_model_returns_subclass(choices: list[IndicatorConfig]) -> None:
    model = build_output_model(choices)
    assert issubclass(model, ClassificationOutputBase)


def test_literal_enforces_enum_in_schema(choices: list[IndicatorConfig]) -> None:
    model = build_output_model(choices)
    schema = model.model_json_schema()
    assert schema["properties"]["indicator_id"]["enum"] == [INDICATOR_A, INDICATOR_B]


def test_valid_indicator_passes(choices: list[IndicatorConfig]) -> None:
    model = build_output_model(choices)
    obj = model.model_validate(_valid_payload(INDICATOR_A))
    assert obj.indicator_id == INDICATOR_A
    assert obj.clause_pattern is ClausePattern.CONDITIONAL_REGIME


def test_invalid_indicator_raises(choices: list[IndicatorConfig]) -> None:
    model = build_output_model(choices)
    with pytest.raises(ValidationError):
        model.model_validate(_valid_payload("9.9"))


def test_missing_evidence_raises(choices: list[IndicatorConfig]) -> None:
    model = build_output_model(choices)
    payload = _valid_payload(INDICATOR_A)
    payload["evidence_element_ids"] = []
    with pytest.raises(ValidationError):
        model.model_validate(payload)


def test_unknown_clause_pattern_raises(choices: list[IndicatorConfig]) -> None:
    model = build_output_model(choices)
    payload = _valid_payload(INDICATOR_A)
    payload["clause_pattern"] = "not_a_pattern"
    with pytest.raises(ValidationError):
        model.model_validate(payload)


def test_extra_fields_rejected(choices: list[IndicatorConfig]) -> None:
    model = build_output_model(choices)
    payload = _valid_payload(INDICATOR_A)
    payload["evil"] = "no"
    with pytest.raises(ValidationError):
        model.model_validate(payload)


def test_empty_decomposition_subject_rejected(choices: list[IndicatorConfig]) -> None:
    model = build_output_model(choices)
    payload = _valid_payload(INDICATOR_A)
    payload["decomposition"] = {
        "subject": "   ",
        "condition": "outbound transfer",
        "constraint": "comparable protection",
    }
    with pytest.raises(ValidationError):
        model.model_validate(payload)


def test_empty_decomposition_constraint_rejected(choices: list[IndicatorConfig]) -> None:
    model = build_output_model(choices)
    payload = _valid_payload(INDICATOR_A)
    payload["decomposition"] = {
        "subject": "organisation",
        "condition": "outbound transfer",
        "constraint": "",
    }
    with pytest.raises(ValidationError):
        model.model_validate(payload)


def test_empty_choices_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        build_output_model([])


def test_dedupes_choice_ids() -> None:
    dup = [
        IndicatorConfig(
            indicator_id=INDICATOR_A,
            name="A",
            definition="A.",
            evaluation_method=EvaluationMethod.CLAUSE_EXTRACTION,
        ),
        IndicatorConfig(
            indicator_id=INDICATOR_A,
            name="A2",
            definition="A.",
            evaluation_method=EvaluationMethod.CLAUSE_EXTRACTION,
        ),
    ]
    model = build_output_model(dup)
    prop = model.model_json_schema()["properties"]["indicator_id"]
    # Pydantic emits `const` for single-value Literal, `enum` for multi.
    assert prop.get("enum") == [INDICATOR_A] or prop.get("const") == INDICATOR_A
