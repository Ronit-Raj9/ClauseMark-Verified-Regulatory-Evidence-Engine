"""Self-consistency / majority-vote tests for the classification service."""

from __future__ import annotations

from collections.abc import Callable

from rie_classify.fake_llm import ScriptedFakeLlm
from rie_classify.service import ClassificationService
from rie_contracts import Element, IndicatorConfig, Layer1Status, PillarConfig

INDICATOR_A = "6.4"
INDICATOR_B = "6.2"


def _payload(indicator_id: str, primary: str, neighbour: str) -> dict[str, object]:
    return {
        "indicator_id": indicator_id,
        "clause_pattern": "conditional_regime",
        "decomposition": {
            "subject": "organisation",
            "condition": "outbound transfer",
            "constraint": "comparable protection",
            "context": None,
        },
        "evidence_element_ids": [primary, neighbour],
        "regime_member_ids": [primary, neighbour],
    }


def test_majority_vote_wins(
    clause_element: Element,
    neighbourhood: list[Element],
    pillar: PillarConfig,
    indicator_choices: list[IndicatorConfig],
    element_store: Callable[[str], Element],
) -> None:
    # 2x A, 1x B → A wins.
    primary = clause_element.element_id
    neighbour = neighbourhood[0].element_id
    llm = ScriptedFakeLlm(
        [
            _payload(INDICATOR_A, primary, neighbour),
            _payload(INDICATOR_A, primary, neighbour),
            _payload(INDICATOR_B, primary, neighbour),
        ]
    )

    svc = ClassificationService(llm, element_store)
    claim = svc.classify_clause(
        clause_element, neighbourhood, pillar, indicator_choices, n_samples=3
    )

    assert claim.indicator_id == INDICATOR_A
    assert claim.self_consistency_votes == {INDICATOR_A: 2, INDICATOR_B: 1}
    assert claim.layer1_status is Layer1Status.PENDING_VERIFICATION


def test_unanimous_vote(
    clause_element: Element,
    neighbourhood: list[Element],
    pillar: PillarConfig,
    indicator_choices: list[IndicatorConfig],
    element_store: Callable[[str], Element],
) -> None:
    primary = clause_element.element_id
    neighbour = neighbourhood[0].element_id
    llm = ScriptedFakeLlm([_payload(INDICATOR_A, primary, neighbour)] * 3)

    svc = ClassificationService(llm, element_store)
    claim = svc.classify_clause(
        clause_element, neighbourhood, pillar, indicator_choices, n_samples=3
    )

    assert claim.indicator_id == INDICATOR_A
    assert claim.self_consistency_votes == {INDICATOR_A: 3}


def test_vote_dict_is_populated_even_when_split(
    clause_element: Element,
    neighbourhood: list[Element],
    pillar: PillarConfig,
    indicator_choices: list[IndicatorConfig],
    element_store: Callable[[str], Element],
) -> None:
    primary = clause_element.element_id
    neighbour = neighbourhood[0].element_id
    # 1 each — Counter.most_common returns first-seen → A wins.
    llm = ScriptedFakeLlm(
        [
            _payload(INDICATOR_A, primary, neighbour),
            _payload(INDICATOR_B, primary, neighbour),
        ]
    )

    svc = ClassificationService(llm, element_store)
    claim = svc.classify_clause(
        clause_element, neighbourhood, pillar, indicator_choices, n_samples=2
    )

    assert claim.self_consistency_votes == {INDICATOR_A: 1, INDICATOR_B: 1}
    assert claim.indicator_id in {INDICATOR_A, INDICATOR_B}


def test_single_sample_works(
    clause_element: Element,
    neighbourhood: list[Element],
    pillar: PillarConfig,
    indicator_choices: list[IndicatorConfig],
    element_store: Callable[[str], Element],
) -> None:
    primary = clause_element.element_id
    neighbour = neighbourhood[0].element_id
    llm = ScriptedFakeLlm([_payload(INDICATOR_A, primary, neighbour)])

    svc = ClassificationService(llm, element_store)
    claim = svc.classify_clause(
        clause_element, neighbourhood, pillar, indicator_choices, n_samples=1
    )

    assert claim.indicator_id == INDICATOR_A
    assert claim.self_consistency_votes == {INDICATOR_A: 1}
