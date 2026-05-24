"""Contract test for `ClassifierPort` — runs against the rie-classify adapter.

The contract: a `ClassifierPort` implementation classifies a clause + its
neighbourhood under a pillar's indicator set and returns a `Claim` that is
- isinstance-compatible with `ClassifierPort`,
- carries `layer1_status = pending_verification`,
- carries a `self_consistency_votes` dict,
- materialises every `EvidenceSpan` from the injected element store (no
  LLM-authored citation strings), and
- on the canonical PDPA cross-border clause picks the indicator whose
  enum value matches the conditional-regime configuration.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

import pytest
from rie_classify import ClassificationService, DeterministicFakeLlm
from rie_contracts import (
    AuthorityTier,
    ClassifierPort,
    ClausePattern,
    DocumentProfile,
    Element,
    ElementType,
    EvaluationMethod,
    IndicatorConfig,
    Layer1Status,
    PillarConfig,
    SpanRole,
)

CANONICAL_64 = "6.4"

CANONICAL_PDPA_TEXT = (
    "An organisation shall not transfer any personal data outside the country "
    "unless it has taken appropriate steps to ensure that the recipient is "
    "bound by legally enforceable obligations to provide to the transferred "
    "personal data a standard of protection that is comparable to the "
    "protection under this Act."
)


@pytest.fixture
def pdpa_clause() -> Element:
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
def pdpa_definition() -> Element:
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
def pdpa_pillar() -> PillarConfig:
    return PillarConfig(
        pillar_id="6",
        pillar_name="Cross-border data transfers",
        cluster="digital_governance",
        document_profile=DocumentProfile.STATUTORY_LEGAL_TEXT,
        status="built",
        description="Restrictions on outbound data transfers.",
        indicators=_indicator_choices(),
    )


def _indicator_choices() -> list[IndicatorConfig]:
    return [
        IndicatorConfig(
            indicator_id="6.1",
            name="Free cross-border data flow",
            definition="The law permits free outbound transfer.",
            evaluation_method=EvaluationMethod.CLAUSE_EXTRACTION,
            clause_pattern=ClausePattern.OBLIGATION,
        ),
        IndicatorConfig(
            indicator_id="6.2",
            name="Prohibition on cross-border transfer",
            definition="The law prohibits outbound transfer.",
            evaluation_method=EvaluationMethod.CLAUSE_EXTRACTION,
            clause_pattern=ClausePattern.PROHIBITION,
        ),
        IndicatorConfig(
            indicator_id=CANONICAL_64,
            name="Conditional cross-border transfer regime",
            definition="Outbound transfer permitted subject to comparable protection.",
            evaluation_method=EvaluationMethod.CLAUSE_EXTRACTION,
            clause_pattern=ClausePattern.CONDITIONAL_REGIME,
            positive_keywords={"en": ["comparable", "unless"]},
            authority_hints=[AuthorityTier.TIER_1_STATUTE],
        ),
    ]


def _element_store(clause: Element, definition: Element) -> Callable[[str], Element]:
    by_id = {clause.element_id: clause, definition.element_id: definition}

    def get(eid: str) -> Element:
        if eid not in by_id:
            raise KeyError(eid)
        return by_id[eid]

    return get


def _service(clause: Element, definition: Element) -> ClassificationService:
    return ClassificationService(
        llm=DeterministicFakeLlm(keyword_to_indicator={("shall not", "comparable"): CANONICAL_64}),
        get_element=_element_store(clause, definition),
        get_jurisdiction=lambda _doc_id: "SAMPLE",
        now=lambda: datetime(2026, 5, 24, tzinfo=UTC),
    )


def test_classifier_is_runtime_isinstance(pdpa_clause: Element, pdpa_definition: Element) -> None:
    svc = _service(pdpa_clause, pdpa_definition)
    assert isinstance(svc, ClassifierPort)


def test_canonical_64_classification(
    pdpa_clause: Element,
    pdpa_definition: Element,
    pdpa_pillar: PillarConfig,
) -> None:
    svc = _service(pdpa_clause, pdpa_definition)
    claim = svc.classify_clause(
        clause_element=pdpa_clause,
        neighbourhood=[pdpa_definition],
        pillar=pdpa_pillar,
        indicator_choices=pdpa_pillar.indicators,
        n_samples=3,
    )

    assert claim.indicator_id == CANONICAL_64
    assert claim.clause_pattern is ClausePattern.CONDITIONAL_REGIME
    assert claim.pillar_id == pdpa_pillar.pillar_id
    assert claim.jurisdiction == "SAMPLE"
    assert claim.layer1_status is Layer1Status.PENDING_VERIFICATION

    # Self-consistency votes recorded.
    assert sum(claim.self_consistency_votes.values()) == 3
    assert claim.self_consistency_votes[CANONICAL_64] >= 2

    # Evidence spans materialised from the element store, never from an
    # LLM-authored citation string.
    primary = [s for s in claim.evidence_spans if s.role is SpanRole.PRIMARY]
    assert len(primary) == 1
    assert primary[0].element_id == pdpa_clause.element_id
    assert primary[0].doc_id == pdpa_clause.doc_id
    assert primary[0].char_start == pdpa_clause.char_start
    assert primary[0].char_end == pdpa_clause.char_end
    assert primary[0].span_id == (
        f"{pdpa_clause.doc_id}#{pdpa_clause.char_start}-{pdpa_clause.char_end}"
    )

    # Regime includes the primary + the linked definition.
    assert claim.regime.primary_element_id == pdpa_clause.element_id
    assert pdpa_definition.element_id in claim.regime.member_element_ids
