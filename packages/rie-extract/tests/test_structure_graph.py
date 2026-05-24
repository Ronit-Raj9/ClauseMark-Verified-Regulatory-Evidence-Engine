"""Unit tests for the deterministic structure-graph builder."""

from __future__ import annotations

from rie_contracts import (
    Element,
    ElementType,
    StructureEdgeType,
)
from rie_extract.structure.graph_builder import StructureGraphBuilder

DOC = "sample_dpa_2020"


def _el(
    eid: str,
    text: str,
    etype: ElementType,
    *,
    parent: str | None = None,
    numbering: str | None = None,
    char_start: int = 0,
) -> Element:
    return Element(
        element_id=eid,
        doc_id=DOC,
        parent_id=parent,
        element_type=etype,
        text=text,
        page=1,
        bbox=None,
        char_start=char_start,
        char_end=char_start + len(text),
        extraction_confidence=1.0,
        legal_numbering=numbering,
    )


def _fixture_elements() -> list[Element]:
    return [
        _el(
            f"{DOC}_s2",
            "Section 2. Definitions.",
            ElementType.SECTION,
            numbering="s. 2",
        ),
        _el(
            f"{DOC}_s2_a",
            '"personal data" means data about an identifiable individual',
            ElementType.DEFINITION,
            parent=f"{DOC}_s2",
            numbering="s. 2(a)",
        ),
        _el(
            f"{DOC}_s5",
            "Section 5. Lawful basis for processing.",
            ElementType.SECTION,
            numbering="s. 5",
            char_start=100,
        ),
        _el(
            f"{DOC}_s5_p1",
            "An organisation shall only process personal data with consent or on another basis specified in section 6.",
            ElementType.PARAGRAPH,
            parent=f"{DOC}_s5",
            char_start=140,
        ),
        _el(
            f"{DOC}_s6",
            "Section 6. Lawful bases other than consent.",
            ElementType.SECTION,
            numbering="s. 6",
            char_start=300,
        ),
        _el(
            f"{DOC}_s26",
            "Section 26. Transfer of personal data outside the country.",
            ElementType.SECTION,
            numbering="s. 26",
            char_start=400,
        ),
        _el(
            f"{DOC}_s261",
            "(1) An organisation shall not transfer any personal data outside the country.",
            ElementType.LIST_ITEM,
            parent=f"{DOC}_s26",
            numbering="s. 26(1)",
            char_start=460,
        ),
        _el(
            f"{DOC}_s262",
            "(2) Subsection (1) does not apply where—",
            ElementType.LIST_ITEM,
            parent=f"{DOC}_s26",
            numbering="s. 26(2)",
            char_start=540,
        ),
        _el(
            f"{DOC}_s27",
            "Section 27. Notwithstanding section 26, the Minister may issue exemptions.",
            ElementType.SECTION,
            numbering="s. 27",
            char_start=600,
        ),
        _el(
            f"{DOC}_s7",
            "Section 7. Provided that the controller has notified the Commissioner.",
            ElementType.PARAGRAPH,
            parent=f"{DOC}_s5",
            char_start=700,
        ),
    ]


def test_cross_reference_edges_resolve_by_numbering() -> None:
    elements = _fixture_elements()
    edges = StructureGraphBuilder().build(elements)
    xrefs = [e for e in edges if e.edge_type == StructureEdgeType.CROSS_REFERENCE]
    assert any(e.from_element == f"{DOC}_s5_p1" and e.to_element == f"{DOC}_s6" for e in xrefs), (
        f"missing s5 → s6 cross-ref; got: {[(e.from_element, e.to_element) for e in xrefs]}"
    )


def test_defines_edges_emitted_from_definitions_section() -> None:
    elements = _fixture_elements()
    edges = StructureGraphBuilder().build(elements)
    defines = [e for e in edges if e.edge_type == StructureEdgeType.DEFINES]
    # The "personal data" definition should link to every element mentioning the term.
    targets = {e.to_element for e in defines if e.from_element == f"{DOC}_s2_a"}
    assert f"{DOC}_s5_p1" in targets, "definition not linked to s5 paragraph"
    assert f"{DOC}_s261" in targets, "definition not linked to s26(1)"


def test_proviso_edges_emitted_for_subsection_exception() -> None:
    elements = _fixture_elements()
    edges = StructureGraphBuilder().build(elements)
    provisos = [e for e in edges if e.edge_type == StructureEdgeType.PROVISO]
    # "(2) Subsection (1) does not apply where—" is a proviso of s. 26.
    assert any(
        e.from_element == f"{DOC}_s262" and e.to_element == f"{DOC}_s26" for e in provisos
    ), f"missing s26(2) proviso; got {[(e.from_element, e.to_element) for e in provisos]}"


def test_proviso_edges_emitted_for_provided_that() -> None:
    elements = _fixture_elements()
    edges = StructureGraphBuilder().build(elements)
    provisos = [e for e in edges if e.edge_type == StructureEdgeType.PROVISO]
    assert any(e.from_element == f"{DOC}_s7" for e in provisos), (
        "missing 'Provided that' proviso edge"
    )


def test_notwithstanding_edges_emitted() -> None:
    elements = _fixture_elements()
    edges = StructureGraphBuilder().build(elements)
    nw = [e for e in edges if e.edge_type == StructureEdgeType.NOTWITHSTANDING]
    assert any(e.from_element == f"{DOC}_s27" and e.to_element == f"{DOC}_s26" for e in nw), (
        f"missing s27 → s26 notwithstanding; got {[(e.from_element, e.to_element) for e in nw]}"
    )


def test_relative_subsection_reference_resolves_within_section() -> None:
    elements = _fixture_elements()
    edges = StructureGraphBuilder().build(elements)
    xrefs = [e for e in edges if e.edge_type == StructureEdgeType.CROSS_REFERENCE]
    # s. 26(2) text says "Subsection (1) does not apply" — should resolve to s. 26(1).
    assert any(e.from_element == f"{DOC}_s262" and e.to_element == f"{DOC}_s261" for e in xrefs), (
        f"relative subsection ref not resolved; got {[(e.from_element, e.to_element) for e in xrefs]}"
    )
