"""Phase-2 integration: structured_tabular post_extract attaches rows + cells.

Feeds a fake TABLE element through `StructuredTabularStrategy.post_extract`
via the public registry and asserts the hierarchy materialises.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

try:
    from rie_contracts import (
        AuthorityTier,
        DocumentMeta,
        DocumentProfile,
        DocumentType,
        Element,
        ElementType,
    )
    from rie_profiles import get_strategy
except ImportError as exc:  # pragma: no cover
    pytest.skip(
        f"rie_profiles / rie_contracts not importable cleanly: {exc}",
        allow_module_level=True,
    )


def _meta() -> DocumentMeta:
    return DocumentMeta(
        doc_id="d_tab",
        jurisdiction="SAMPLE",
        title="Tariff Schedule",
        document_type=DocumentType.STATUTE,
        authority_tier=AuthorityTier.TIER_1_STATUTE,
        sha256="b" * 64,
        retrieved_at=datetime.now(tz=UTC),
    )


def _table(text: str) -> Element:
    return Element(
        element_id="tbl1",
        doc_id="d_tab",
        element_type=ElementType.TABLE,
        text=text,
        page=1,
        char_start=0,
        char_end=len(text),
        extraction_confidence=0.95,
    )


@pytest.mark.integration
def test_structured_tabular_post_extract_attaches_rows_and_cells() -> None:
    strat = get_strategy(DocumentProfile.STRUCTURED_TABULAR)
    table = _table("HS\tDescription\tRate\n0101\tHorses\t5%\n0102\tCattle\t0%")
    elements, edges = strat.post_extract(_meta(), [table], [])

    rows = [e for e in elements if e.legal_numbering and e.legal_numbering.startswith("row:")
            and "col:" not in e.legal_numbering]
    cells = [e for e in elements if e.element_type is ElementType.TABLE_CELL]

    # 3 lines → 3 rows; 3 cells each → 9 cells.
    assert len(rows) == 3
    assert len(cells) == 9

    # Rows parented to the table.
    assert all(r.parent_id == "tbl1" for r in rows)
    # Cells parented to their row.
    for cell in cells:
        assert cell.parent_id is not None
        assert cell.parent_id.startswith("tbl1.r")

    # Original table preserved in output.
    assert any(e.element_id == "tbl1" for e in elements)
    # No edges fabricated.
    assert list(edges) == []


@pytest.mark.integration
def test_structured_tabular_respects_preexisting_children() -> None:
    strat = get_strategy(DocumentProfile.STRUCTURED_TABULAR)
    table = _table("a|b\nc|d")
    pre = Element(
        element_id="tbl1.pre",
        doc_id="d_tab",
        parent_id="tbl1",
        element_type=ElementType.TABLE_CELL,
        text="a",
        page=1,
        char_start=0,
        char_end=1,
        extraction_confidence=1.0,
    )
    out, _ = strat.post_extract(_meta(), [table, pre], [])
    # Strategy must NOT split when children already attached upstream.
    assert len(out) == 2
