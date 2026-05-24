from datetime import UTC, datetime

import pytest
from rie_contracts import (
    AuthorityTier,
    BoundingBox,
    DocumentMeta,
    DocumentType,
    Element,
    ElementType,
    EvidenceSpan,
    GateName,
    GateResult,
    StructureEdge,
    StructureEdgeType,
    VerificationStatus,
)
from rie_domain import (
    AuthorityResolver,
    assemble_regime,
    build_citation,
    consolidate_self_consistency_votes,
    derive_status,
    render_citation_string,
)
from rie_domain.authority import AuthorityOverride


def _t() -> datetime:
    return datetime(2026, 5, 24, tzinfo=UTC)


def test_authority_override_elevates() -> None:
    resolver = AuthorityResolver(
        overrides=[
            AuthorityOverride(
                jurisdiction="CIV",
                source_pattern=r"circular_.*",
                tier=AuthorityTier.TIER_1_STATUTE,
                rationale="regulator circulars binding in CIV",
            )
        ]
    )
    out = resolver.resolve("CIV", "circular_2024_05", None, AuthorityTier.TIER_3_GUIDELINE)
    assert out == AuthorityTier.TIER_1_STATUTE


def test_authority_no_match() -> None:
    resolver = AuthorityResolver()
    assert (
        resolver.resolve("X", "src", None, AuthorityTier.TIER_2_REGULATION)
        == AuthorityTier.TIER_2_REGULATION
    )


def test_assemble_regime_walks_provisos_and_definitions() -> None:
    elements = {
        eid: Element(
            element_id=eid,
            doc_id="d1",
            element_type=ElementType.PARAGRAPH,
            text="x",
            page=1,
            char_start=0,
            char_end=1,
            extraction_confidence=1.0,
        )
        for eid in ("s26", "s2", "s26_2", "remote")
    }
    edges = [
        StructureEdge(from_element="s26", to_element="s2", edge_type=StructureEdgeType.DEFINES),
        StructureEdge(from_element="s26", to_element="s26_2", edge_type=StructureEdgeType.PROVISO),
        StructureEdge(
            from_element="remote", to_element="s26", edge_type=StructureEdgeType.CROSS_REFERENCE
        ),
    ]
    regime = assemble_regime(elements["s26"], elements, edges, max_depth=1)
    assert "s2" in regime.definitions
    assert "s26_2" in regime.exceptions
    assert "remote" in regime.member_element_ids
    assert regime.primary_element_id == "s26"


def test_consolidate_votes_majority() -> None:
    label, counts = consolidate_self_consistency_votes(["6.4", "6.4", "6.3"])
    assert label == "6.4"
    assert counts == {"6.4": 2, "6.3": 1}


def test_consolidate_votes_empty() -> None:
    label, counts = consolidate_self_consistency_votes([])
    assert label == ""
    assert counts == {}


def test_derive_status_all_pass() -> None:
    gates = [GateResult(gate=g, passed=True, ran_at=_t()) for g in GateName]
    status, failures = derive_status(gates)
    assert status == VerificationStatus.VERIFIED
    assert failures == []


def test_derive_status_deterministic_fail_is_rejected() -> None:
    gates = [
        GateResult(gate=GateName.SPAN_EXISTENCE, passed=False, detail="not found", ran_at=_t()),
        GateResult(gate=GateName.VERBATIM_MATCH, passed=True, ran_at=_t()),
        GateResult(gate=GateName.ENTAILMENT, passed=True, ran_at=_t()),
        GateResult(gate=GateName.SELF_CONSISTENCY, passed=True, ran_at=_t()),
    ]
    status, failures = derive_status(gates)
    assert status == VerificationStatus.REJECTED
    assert any("span_existence" in f for f in failures)


def test_derive_status_model_disagreement_is_flagged() -> None:
    gates = [
        GateResult(gate=GateName.SPAN_EXISTENCE, passed=True, ran_at=_t()),
        GateResult(gate=GateName.VERBATIM_MATCH, passed=True, ran_at=_t()),
        GateResult(gate=GateName.ENTAILMENT, passed=False, detail="NLI/LLM disagree", ran_at=_t()),
        GateResult(gate=GateName.SELF_CONSISTENCY, passed=True, ran_at=_t()),
    ]
    status, _ = derive_status(gates)
    assert status == VerificationStatus.FLAGGED


def test_build_citation_renders_locator() -> None:
    doc = DocumentMeta(
        doc_id="d1",
        jurisdiction="SAMPLE",
        title="Test Act",
        document_type=DocumentType.STATUTE,
        authority_tier=AuthorityTier.TIER_1_STATUTE,
        sha256="a" * 64,
        retrieved_at=_t(),
        effective_date=_t(),
    )
    element = Element(
        element_id="e1",
        doc_id="d1",
        element_type=ElementType.PARAGRAPH,
        text="hi",
        page=4,
        bbox=BoundingBox(page=4, x0=0, y0=0, x1=1, y1=1),
        char_start=10,
        char_end=12,
        extraction_confidence=1.0,
        legal_numbering="Art. 26(1)",
    )
    span = EvidenceSpan(
        span_id="d1#10-12",
        element_id="e1",
        doc_id="d1",
        char_start=10,
        char_end=12,
    )
    c = build_citation(span, element, doc, "hi")
    rendered = render_citation_string(c)
    assert "Test Act" in rendered
    assert "Art. 26(1)" in rendered
    assert "p. 4" in rendered


def test_build_citation_doc_mismatch_raises() -> None:
    doc = DocumentMeta(
        doc_id="d1",
        jurisdiction="SAMPLE",
        title="t",
        document_type=DocumentType.STATUTE,
        authority_tier=AuthorityTier.TIER_1_STATUTE,
        sha256="a" * 64,
        retrieved_at=_t(),
    )
    element = Element(
        element_id="e1",
        doc_id="OTHER",
        element_type=ElementType.PARAGRAPH,
        text="x",
        page=1,
        char_start=0,
        char_end=1,
        extraction_confidence=1.0,
    )
    span = EvidenceSpan(
        span_id="OTHER#0-1", element_id="e1", doc_id="OTHER", char_start=0, char_end=1
    )
    with pytest.raises(AssertionError):
        build_citation(span, element, doc, "x")
