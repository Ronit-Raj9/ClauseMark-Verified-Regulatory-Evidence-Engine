"""Unit tests for the heuristic entity extractor and the KG-grounding gate.

The extractor is heuristic by design — these tests pin the supported
patterns documented in ``rie_verify.kg_grounding``; they do NOT claim
the extractor is exhaustive.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, datetime

import pytest
from rie_contracts import (
    Claim,
    ClausePattern,
    Decomposition,
    EvidenceSpan,
    Layer1Status,
    LegalRegime,
    VerificationStatus,
)
from rie_verify import (
    FakeNliBackend,
    FakeSecondLlm,
    VerificationService,
)
from rie_verify.kg_grounding import (
    KG_GATE_NAME,
    EntityKind,
    build_document_kg,
    extract_entities,
    run_kg_grounding_gate,
)

# ──────────────────────────────────────────────────────────────────────────
# Entity extractor — defined-terms
# ──────────────────────────────────────────────────────────────────────────


def test_extract_defined_term_means() -> None:
    text = '"personal data" means any information relating to an identified person.'
    entities = extract_entities(text)
    kinds = {(e.kind, e.key) for e in entities}
    assert (EntityKind.DEFINED_TERM, "personal data") in kinds


def test_extract_defined_term_shall_mean() -> None:
    text = '"data controller" shall mean a person who determines the purposes.'
    entities = extract_entities(text)
    assert any(e.kind is EntityKind.DEFINED_TERM and e.key == "data controller" for e in entities)


def test_extract_defined_term_refers_to() -> None:
    text = '"cross-border transfer" refers to the movement of data abroad.'
    entities = extract_entities(text)
    assert any(
        e.kind is EntityKind.DEFINED_TERM and e.key == "cross-border transfer" for e in entities
    )


def test_extract_no_defined_term_without_marker() -> None:
    text = '"personal data" is important to safeguard.'  # no shall mean / means / refers to
    entities = extract_entities(text)
    assert not any(e.kind is EntityKind.DEFINED_TERM for e in entities)


# ──────────────────────────────────────────────────────────────────────────
# Entity extractor — cross-references
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "phrase",
    [
        "section 26",
        "Section 26(2)",
        "article 4",
        "Article 4(1)",
        "paragraph 1",
        "Paragraph 1(a)",
        "sub-section 3",
        "sub-paragraph 2",
        "section 26A",
    ],
)
def test_extract_cross_reference(phrase: str) -> None:
    text = f"As provided in {phrase}, the transferor shall comply."
    entities = extract_entities(text)
    assert any(e.kind is EntityKind.CROSS_REFERENCE for e in entities), (
        f"expected cross_reference for {phrase!r}, got {entities}"
    )


def test_extract_multiple_cross_references() -> None:
    text = "See section 26(2) and Article 4(1)."
    entities = extract_entities(text)
    crefs = [e for e in entities if e.kind is EntityKind.CROSS_REFERENCE]
    assert len(crefs) == 2


# ──────────────────────────────────────────────────────────────────────────
# Entity extractor — dates
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "phrase",
    [
        "2020-01-01",
        "1 January 2020",
        "January 1, 2020",
        "1st January 2020",
        "31 December 2024",
    ],
)
def test_extract_date(phrase: str) -> None:
    text = f"The Act commenced on {phrase}."
    entities = extract_entities(text)
    assert any(e.kind is EntityKind.DATE for e in entities), (
        f"expected date for {phrase!r}, got {entities}"
    )


def test_extract_invalid_iso_date_rejected() -> None:
    text = "The pseudo-date 2020-13-40 is not valid."
    entities = extract_entities(text)
    assert not any(e.kind is EntityKind.DATE for e in entities)


# ──────────────────────────────────────────────────────────────────────────
# Entity extractor — money
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "phrase",
    ["USD 1,000,000", "$50", "EUR 250.75", "£100", "INR 5,000", "¥1000"],
)
def test_extract_money(phrase: str) -> None:
    text = f"A penalty of {phrase} may be imposed."
    entities = extract_entities(text)
    assert any(e.kind is EntityKind.MONEY for e in entities), (
        f"expected money for {phrase!r}, got {entities}"
    )


# ──────────────────────────────────────────────────────────────────────────
# Extractor de-dup
# ──────────────────────────────────────────────────────────────────────────


def test_extract_dedupes_within_call() -> None:
    text = "See section 26. See section 26 again. See SECTION 26 once more."
    entities = extract_entities(text)
    crefs = [e for e in entities if e.kind is EntityKind.CROSS_REFERENCE]
    assert len(crefs) == 1


# ──────────────────────────────────────────────────────────────────────────
# Document KG
# ──────────────────────────────────────────────────────────────────────────


def test_build_document_kg_indexes_all_kinds() -> None:
    elements = {
        "el_1": '"personal data" means any information about a person.',
        "el_2": "See section 26 effective 2020-01-01. Fine: USD 1,000,000.",
    }
    kg = build_document_kg(elements)
    assert kg.has("personal data")
    assert kg.has("section 26")
    assert kg.has("2020-01-01")
    assert kg.has("USD 1,000,000")
    assert not kg.has("nonexistent term")


def test_build_document_kg_cooccurrence_edges() -> None:
    elements = {
        "el_1": "See section 26 effective 2020-01-01.",
    }
    kg = build_document_kg(elements)
    # section 26 and 2020-01-01 co-occur in the same element
    assert "2020-01-01" in kg.edges["section 26"]
    assert "section 26" in kg.edges["2020-01-01"]


def test_build_document_kg_parent_cooccurrence() -> None:
    elements = {
        "parent": '"data controller" means a determining person.',
        "child": "See section 26.",
    }
    parent_of: dict[str, str | None] = {"child": "parent", "parent": None}
    kg = build_document_kg(elements, parent_of=parent_of)
    # child + parent entities co-locate via the structure-graph edge
    assert "data controller" in kg.edges["section 26"]


# ──────────────────────────────────────────────────────────────────────────
# Gate behaviour
# ──────────────────────────────────────────────────────────────────────────


def _build_claim(
    *,
    decomposition: Decomposition,
    element_id: str,
    element_text_len: int,
) -> Claim:
    span = EvidenceSpan(
        span_id=f"doc#0-{element_text_len}",
        element_id=element_id,
        doc_id="doc",
        char_start=0,
        char_end=element_text_len,
    )
    return Claim(
        claim_id="claim_kg_001",
        indicator_id="6.4",
        pillar_id="6",
        clause_id=element_id,
        jurisdiction="SAMPLE",
        clause_pattern=ClausePattern.OBLIGATION,
        decomposition=decomposition,
        evidence_spans=[span],
        regime=LegalRegime(primary_element_id=element_id, member_element_ids=[element_id]),
        layer1_status=Layer1Status.PENDING_VERIFICATION,
        self_consistency_votes={"6.4": 3},
        created_at=datetime(2026, 5, 24, tzinfo=UTC),
    )


def test_kg_gate_passes_when_all_entities_resolve() -> None:
    element_id = "el_pass"
    text = (
        '"data controller" means a person who determines the purposes. '
        "See section 26 effective 2020-01-01."
    )

    store = {element_id: text}

    def resolver(element_id: str) -> str:
        return store[element_id]

    claim = _build_claim(
        decomposition=Decomposition(
            subject='A "data controller"',
            constraint="comply with section 26 by 2020-01-01",
        ),
        element_id=element_id,
        element_text_len=len(text),
    )
    result = run_kg_grounding_gate(claim, resolver)
    assert result.gate == KG_GATE_NAME
    assert result.passed is True
    assert result.flagged is False
    assert result.unresolved == ()


def test_kg_gate_flags_when_entity_missing() -> None:
    element_id = "el_flag"
    text = '"data controller" means a person who determines the purposes.'

    store = {element_id: text}

    def resolver(element_id: str) -> str:
        return store[element_id]

    claim = _build_claim(
        decomposition=Decomposition(
            subject='A "data controller"',
            constraint="comply with section 99 by 2099-12-31",
        ),
        element_id=element_id,
        element_text_len=len(text),
    )
    result = run_kg_grounding_gate(claim, resolver)
    assert result.passed is False
    assert result.flagged is True
    assert any("section 99" in u for u in result.unresolved)
    assert any("2099-12-31" in u for u in result.unresolved)


def test_kg_gate_vacuous_pass_when_decomposition_has_no_entities() -> None:
    element_id = "el_vac"
    text = "An organisation shall act with reasonable care."

    store = {element_id: text}

    def resolver(element_id: str) -> str:
        return store[element_id]

    claim = _build_claim(
        decomposition=Decomposition(
            subject="An organisation",
            constraint="act with reasonable care",
        ),
        element_id=element_id,
        element_text_len=len(text),
    )
    result = run_kg_grounding_gate(claim, resolver)
    assert result.passed is True
    assert "vacuous" in result.detail


def test_kg_gate_flags_when_element_unresolvable() -> None:
    element_id = "el_missing"

    def resolver(element_id: str) -> str:
        raise KeyError(element_id)

    claim = _build_claim(
        decomposition=Decomposition(subject="x", constraint="y"),
        element_id=element_id,
        element_text_len=1,
    )
    result = run_kg_grounding_gate(claim, resolver)
    assert result.passed is False
    assert result.flagged is True


# ──────────────────────────────────────────────────────────────────────────
# Service wiring — flag default-off, opt-in plumbing
# ──────────────────────────────────────────────────────────────────────────


def test_kg_gate_disabled_by_default(
    claim_factory: Callable[..., Claim],
    resolver: Callable[[str], str],
) -> None:
    svc = VerificationService(
        nli_model=FakeNliBackend(entailment_prob=0.9, contradiction_prob=0.05, neutral_prob=0.05),
        second_llm=FakeSecondLlm(verdict=True),
    )
    assert svc.kg_gate_enabled is False
    claim = claim_factory(self_consistency_votes={"6.4": 3})
    report, kg = svc.verify_with_kg(claim, resolver)
    assert kg is None
    assert report.status is VerificationStatus.VERIFIED
    # Canonical contract preserved.
    assert len(report.gates) == 4


def test_kg_gate_enabled_returns_kg_result(
    claim_factory: Callable[..., Claim],
    resolver: Callable[[str], str],
) -> None:
    svc = VerificationService(
        nli_model=FakeNliBackend(entailment_prob=0.9, contradiction_prob=0.05, neutral_prob=0.05),
        second_llm=FakeSecondLlm(verdict=True),
        kg_gate_enabled=True,
    )
    claim = claim_factory(self_consistency_votes={"6.4": 3})
    report, kg = svc.verify_with_kg(claim, resolver)
    assert kg is not None
    assert kg.gate == KG_GATE_NAME
    # The default fixture decomposition has no extractable entities → vacuous pass.
    assert kg.passed is True
    assert report.status is VerificationStatus.VERIFIED


def test_kg_gate_flag_downgrades_verified_to_flagged(
    text_store: dict[str, str],
) -> None:
    element_id = "doc_el_1"
    element_text = (
        '"personal data" means information about a person. See section 26 effective 2020-01-01.'
    )
    text_store_local: Mapping[str, str] = {element_id: element_text}

    def resolver(element_id: str) -> str:
        return text_store_local[element_id]

    svc = VerificationService(
        nli_model=FakeNliBackend(entailment_prob=0.9, contradiction_prob=0.05, neutral_prob=0.05),
        second_llm=FakeSecondLlm(verdict=True),
        kg_gate_enabled=True,
    )
    claim = _build_claim(
        decomposition=Decomposition(
            subject="An organisation",
            constraint="comply with section 99 — not present in the document",
        ),
        element_id=element_id,
        element_text_len=len(element_text),
    )
    report, kg = svc.verify_with_kg(claim, resolver)
    assert kg is not None
    assert kg.flagged is True
    # Canonical 4 gates all pass → would have been VERIFIED; KG flag downgrades.
    assert report.status is VerificationStatus.FLAGGED
    assert any(KG_GATE_NAME in r for r in report.failure_reasons)
    # The contract object still has exactly 4 canonical gates.
    assert len(report.gates) == 4


def test_kg_gate_does_not_resurrect_rejected(
    claim_factory: Callable[..., Claim],
    resolver: Callable[[str], str],
    element_text: str,
) -> None:
    """A REJECTED claim stays REJECTED — KG is not consulted."""
    svc = VerificationService(
        nli_model=FakeNliBackend(entailment_prob=0.9, contradiction_prob=0.05, neutral_prob=0.05),
        second_llm=FakeSecondLlm(verdict=True),
        kg_gate_enabled=True,
    )
    overshoot = len(element_text) + 100
    bad = EvidenceSpan(
        span_id=f"sample_dpa_2020#0-{overshoot}",
        element_id="sample_dpa_2020_s26_p1",
        doc_id="sample_dpa_2020",
        char_start=0,
        char_end=overshoot,
    )
    claim = claim_factory(evidence_spans=[bad], self_consistency_votes={"6.4": 3})
    report, kg = svc.verify_with_kg(claim, resolver)
    assert report.status is VerificationStatus.REJECTED
    assert kg is None  # skipped because deterministic gates failed
