"""Unit tests for Phase-2 post_extract behaviour."""

from __future__ import annotations

from datetime import UTC, datetime

from rie_contracts import (
    AuthorityTier,
    DocumentMeta,
    DocumentProfile,
    DocumentType,
    Element,
    ElementType,
    StructureEdge,
    StructureEdgeType,
)
from rie_profiles import get_strategy


def _meta(jurisdiction: str = "SAMPLE", doc_id: str = "d1") -> DocumentMeta:
    return DocumentMeta(
        doc_id=doc_id,
        jurisdiction=jurisdiction,
        title="Test Doc",
        document_type=DocumentType.STATUTE,
        authority_tier=AuthorityTier.TIER_1_STATUTE,
        sha256="a" * 64,
        retrieved_at=datetime.now(tz=UTC),
    )


def _table_element(text: str, *, char_start: int = 0) -> Element:
    return Element(
        element_id="t1",
        doc_id="d1",
        element_type=ElementType.TABLE,
        text=text,
        page=1,
        char_start=char_start,
        char_end=char_start + len(text),
        extraction_confidence=0.9,
    )


# ─── structured_tabular ────────────────────────────────────────────────────


def test_structured_tabular_splits_table_into_rows_and_cells() -> None:
    strat = get_strategy(DocumentProfile.STRUCTURED_TABULAR)
    table = _table_element("HS\tDescription\tRate\n0101\tLive horses\t5%")
    out_elements, out_edges = strat.post_extract(_meta(), [table], [])

    rows = [e for e in out_elements if e.element_type is ElementType.OTHER]
    cells = [e for e in out_elements if e.element_type is ElementType.TABLE_CELL]
    assert len(rows) == 2
    assert len(cells) == 6
    assert all(r.parent_id == "t1" for r in rows)
    # Cells parented to their row.
    for cell in cells:
        assert cell.parent_id is not None
        assert cell.parent_id.startswith("t1.r")
    # Original table preserved.
    assert any(e.element_id == "t1" for e in out_elements)
    # No edges fabricated (StructureEdgeType has no CONTAINS).
    assert list(out_edges) == []


def test_structured_tabular_respects_existing_children() -> None:
    strat = get_strategy(DocumentProfile.STRUCTURED_TABULAR)
    table = _table_element("a|b\nc|d")
    pre_child = Element(
        element_id="t1.pre",
        doc_id="d1",
        parent_id="t1",
        element_type=ElementType.TABLE_CELL,
        text="a",
        page=1,
        char_start=0,
        char_end=1,
        extraction_confidence=1.0,
    )
    out, _ = strat.post_extract(_meta(), [table, pre_child], [])
    # Should not add new row/cell elements when a child is already attached.
    assert len(out) == 2


def test_structured_tabular_skips_llm() -> None:
    strat = get_strategy(DocumentProfile.STRUCTURED_TABULAR)
    assert strat.supports_llm_classification() is False
    assert strat.child_chunk_size() == 256
    assert ElementType.TABLE.value in strat.applicable_element_types()


def test_structured_tabular_single_column_row() -> None:
    strat = get_strategy(DocumentProfile.STRUCTURED_TABULAR)
    table = _table_element("only one column")
    out, _ = strat.post_extract(_meta(), [table], [])
    cells = [e for e in out if e.element_type is ElementType.TABLE_CELL]
    assert len(cells) == 1
    assert cells[0].text == "only one column"


# ─── mixed_regulatory ──────────────────────────────────────────────────────


def test_mixed_regulatory_routes_prose_to_llm() -> None:
    strat = get_strategy(DocumentProfile.MIXED_REGULATORY)
    para = Element(
        element_id="p1",
        doc_id="d1",
        element_type=ElementType.PARAGRAPH,
        text="An obligation shall apply.",
        page=1,
        char_start=0,
        char_end=27,
        extraction_confidence=1.0,
    )
    out, _ = strat.post_extract(_meta(), [para], [])
    assert out[0].legal_numbering is not None
    assert "route:clause_extraction" in out[0].legal_numbering


def test_mixed_regulatory_routes_tables_to_tabular() -> None:
    strat = get_strategy(DocumentProfile.MIXED_REGULATORY)
    table = _table_element("HS\tRate\n01\t5%")
    out, _ = strat.post_extract(_meta(), [table], [])
    assert out[0].legal_numbering is not None
    assert "route:tabular_lookup" in out[0].legal_numbering


def test_mixed_regulatory_preserves_existing_numbering() -> None:
    strat = get_strategy(DocumentProfile.MIXED_REGULATORY)
    art = Element(
        element_id="a1",
        doc_id="d1",
        element_type=ElementType.ARTICLE,
        text="Art 1 text",
        page=1,
        char_start=0,
        char_end=10,
        extraction_confidence=1.0,
        legal_numbering="Art. 1",
    )
    out, _ = strat.post_extract(_meta(), [art], [])
    assert out[0].legal_numbering is not None
    assert "Art. 1" in out[0].legal_numbering
    assert "route:clause_extraction" in out[0].legal_numbering


def test_mixed_regulatory_skips_unknown_types() -> None:
    strat = get_strategy(DocumentProfile.MIXED_REGULATORY)
    other = Element(
        element_id="o1",
        doc_id="d1",
        element_type=ElementType.HEADING,
        text="Header",
        page=1,
        char_start=0,
        char_end=6,
        extraction_confidence=1.0,
    )
    out, _ = strat.post_extract(_meta(), [other], [])
    # HEADING isn't in either route map → untouched.
    assert out[0].legal_numbering is None


def test_mixed_regulatory_keeps_edges() -> None:
    strat = get_strategy(DocumentProfile.MIXED_REGULATORY)
    edge = StructureEdge(
        from_element="p1",
        to_element="p2",
        edge_type=StructureEdgeType.CROSS_REFERENCE,
    )
    _, out_edges = strat.post_extract(_meta(), [], [edge])
    assert list(out_edges) == [edge]


# ─── treaty_membership ────────────────────────────────────────────────────


def test_treaty_membership_collapses_to_single_row() -> None:
    strat = get_strategy(DocumentProfile.TREATY_MEMBERSHIP)
    el = Element(
        element_id="x1",
        doc_id="d1",
        element_type=ElementType.PARAGRAPH,
        text="Country X acceded on 2020-01-01.",
        page=1,
        char_start=0,
        char_end=32,
        extraction_confidence=1.0,
    )
    out, out_edges = strat.post_extract(_meta(jurisdiction="SAMPLE"), [el], [])
    assert len(out) == 1
    row = out[0]
    assert row.element_type is ElementType.OTHER
    assert row.element_id == "d1#membership:SAMPLE"
    assert row.legal_numbering == "membership:SAMPLE"
    assert "Country X acceded" in row.text
    assert list(out_edges) == []


def test_treaty_membership_falls_back_to_title_when_empty() -> None:
    strat = get_strategy(DocumentProfile.TREATY_MEMBERSHIP)
    out, _ = strat.post_extract(_meta(), [], [])
    assert len(out) == 1
    assert out[0].text == "Test Doc"


def test_treaty_membership_drops_edges() -> None:
    strat = get_strategy(DocumentProfile.TREATY_MEMBERSHIP)
    edge = StructureEdge(
        from_element="a",
        to_element="b",
        edge_type=StructureEdgeType.CROSS_REFERENCE,
    )
    _, out_edges = strat.post_extract(_meta(), [], [edge])
    assert list(out_edges) == []


def test_treaty_membership_no_llm_no_chunking() -> None:
    strat = get_strategy(DocumentProfile.TREATY_MEMBERSHIP)
    assert strat.supports_llm_classification() is False
    assert strat.child_chunk_size() == 0
