"""Shared fixtures for rie-classify tests."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

import pytest
from rie_contracts import (
    ClausePattern,
    DocumentProfile,
    Element,
    ElementType,
    EvaluationMethod,
    IndicatorConfig,
    PillarConfig,
)

# Canonical canonical-PDPA clause text — used in both the service test and
# the contract test. Lives in tests/, not src/, so it stays out of engine
# code and does not trip the `test_no_hardcoded_indicator_ids` fitness fn.
CANONICAL_PDPA_TEXT = (
    "An organisation shall not transfer any personal data outside the country "
    "unless it has taken appropriate steps to ensure that the recipient is "
    "bound by legally enforceable obligations to provide to the transferred "
    "personal data a standard of protection that is comparable to the "
    "protection under this Act."
)


@pytest.fixture
def now() -> datetime:
    return datetime(2026, 5, 24, tzinfo=UTC)


@pytest.fixture
def clause_element() -> Element:
    return Element(
        element_id="sample_dpa_2020_s26_p1",
        doc_id="sample_dpa_2020",
        parent_id="sample_dpa_2020_s26",
        element_type=ElementType.PARAGRAPH,
        text=CANONICAL_PDPA_TEXT,
        page=4,
        char_start=4120,
        char_end=4120 + len(CANONICAL_PDPA_TEXT),
        extraction_confidence=0.99,
        legal_numbering="s26(1)",
    )


@pytest.fixture
def definition_element() -> Element:
    text = (
        '"personal data" means data, whether true or not, about an individual '
        "who can be identified from that data."
    )
    return Element(
        element_id="sample_dpa_2020_s2_def_personal_data",
        doc_id="sample_dpa_2020",
        parent_id="sample_dpa_2020_s2",
        element_type=ElementType.DEFINITION,
        text=text,
        page=1,
        char_start=200,
        char_end=200 + len(text),
        extraction_confidence=0.99,
        legal_numbering="s2",
    )


@pytest.fixture
def neighbourhood(definition_element: Element) -> list[Element]:
    return [definition_element]


@pytest.fixture
def indicator_choices() -> list[IndicatorConfig]:
    return [
        IndicatorConfig(
            indicator_id="6.1",
            name="Free cross-border data flow",
            definition="Whether the law permits free outbound transfer.",
            evaluation_method=EvaluationMethod.CLAUSE_EXTRACTION,
            clause_pattern=ClausePattern.OBLIGATION,
            scoring_criteria={"0": "prohibited", "1": "free"},
            positive_keywords={"en": ["may transfer", "no restriction"]},
        ),
        IndicatorConfig(
            indicator_id="6.2",
            name="Prohibition on cross-border transfer",
            definition="Whether the law prohibits outbound transfer.",
            evaluation_method=EvaluationMethod.CLAUSE_EXTRACTION,
            clause_pattern=ClausePattern.PROHIBITION,
            positive_keywords={"en": ["shall not transfer", "prohibited"]},
        ),
        IndicatorConfig(
            indicator_id="6.4",
            name="Conditional cross-border transfer regime",
            definition=(
                "Whether the law permits outbound transfer subject to "
                "comparable-protection / adequacy / safeguards."
            ),
            evaluation_method=EvaluationMethod.CLAUSE_EXTRACTION,
            clause_pattern=ClausePattern.CONDITIONAL_REGIME,
            positive_keywords={
                "en": ["comparable", "adequacy", "appropriate safeguards", "unless"],
            },
            few_shot_examples=[
                {
                    "text": CANONICAL_PDPA_TEXT,
                    "label": "6.4",
                    "rationale": "general rule + comparable-protection exception",
                }
            ],
        ),
    ]


@pytest.fixture
def pillar(indicator_choices: list[IndicatorConfig]) -> PillarConfig:
    return PillarConfig(
        pillar_id="6",
        pillar_name="Cross-border data transfers",
        cluster="digital_governance",
        document_profile=DocumentProfile.STATUTORY_LEGAL_TEXT,
        status="built",
        description="Restrictions and conditions on outbound transfers.",
        indicators=indicator_choices,
    )


@pytest.fixture
def element_store(clause_element: Element, definition_element: Element) -> Callable[[str], Element]:
    by_id = {
        clause_element.element_id: clause_element,
        definition_element.element_id: definition_element,
    }

    def get(eid: str) -> Element:
        if eid not in by_id:
            raise KeyError(eid)
        return by_id[eid]

    return get
