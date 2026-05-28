"""Regime assembly helpers for §6.4 — graph walk is injected, LLM hints merged here.

``rie-classify`` cannot import ``rie_domain`` (architecture fitness function).
Graph traversal via ``assemble_regime`` lives in orchestration/domain; this
module defines the structured hand-off from constrained decoding and helpers
to merge LLM-emitted ``regime_member_ids`` into an assembled ``LegalRegime``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from rie_contracts import Element, ElementType, LegalRegime

RegimeAssembler = Callable[[Element, Sequence[str]], LegalRegime]
LegacyRegimeAssembler = Callable[[Element], LegalRegime]


@dataclass(frozen=True, slots=True)
class RegimeAssemblyInput:
    """Structured ids for orchestration/domain graph-walk assembly (§6.4)."""

    primary_element_id: str
    llm_regime_member_ids: tuple[str, ...]
    doc_id: str


def build_regime_assembly_input(
    clause: Element,
    llm_regime_member_ids: Sequence[str],
) -> RegimeAssemblyInput:
    """Package LLM regime hints for an injected graph-walk assembler."""

    return RegimeAssemblyInput(
        primary_element_id=clause.element_id,
        llm_regime_member_ids=tuple(llm_regime_member_ids),
        doc_id=clause.doc_id,
    )


def invoke_regime_assembler(
    assembler: RegimeAssembler | LegacyRegimeAssembler,
    *,
    clause: Element,
    llm_regime_member_ids: Sequence[str],
) -> LegalRegime:
    """Call a wired graph-walk assembler; tolerate legacy single-arg wiring."""

    try:
        return assembler(clause, llm_regime_member_ids)  # type: ignore[call-arg]
    except TypeError:
        return assembler(clause)  # type: ignore[call-arg]


def materialise_regime_from_llm_ids(
    *,
    primary_element_id: str,
    regime_member_ids: Sequence[str],
    get_element: Callable[[str], Element],
) -> LegalRegime:
    """Fallback regime when no graph-walk assembler is wired."""

    members: list[str] = [primary_element_id]
    definitions: list[str] = []
    seen: set[str] = {primary_element_id}

    for eid in regime_member_ids:
        if eid in seen:
            continue
        try:
            el = get_element(eid)
        except KeyError:
            continue
        seen.add(eid)
        members.append(eid)
        if el.element_type is ElementType.DEFINITION:
            definitions.append(eid)

    return LegalRegime(
        primary_element_id=primary_element_id,
        member_element_ids=members,
        definitions=definitions,
    )


def merge_llm_regime_hints(
    graph_regime: LegalRegime,
    llm_regime_member_ids: Sequence[str],
    *,
    get_element: Callable[[str], Element],
) -> LegalRegime:
    """Union graph-walk members with LLM-emitted regime_member_ids."""

    members = list(graph_regime.member_element_ids)
    definitions = list(graph_regime.definitions)
    exceptions = list(graph_regime.exceptions)
    seen = set(members)

    for eid in llm_regime_member_ids:
        if eid in seen:
            continue
        try:
            el = get_element(eid)
        except KeyError:
            continue
        seen.add(eid)
        members.append(eid)
        if el.element_type is ElementType.DEFINITION and eid not in definitions:
            definitions.append(eid)

    return graph_regime.model_copy(
        update={
            "member_element_ids": members,
            "definitions": definitions,
            "exceptions": exceptions,
        }
    )


__all__ = [
    "LegacyRegimeAssembler",
    "RegimeAssembler",
    "RegimeAssemblyInput",
    "build_regime_assembly_input",
    "invoke_regime_assembler",
    "materialise_regime_from_llm_ids",
    "merge_llm_regime_hints",
]
