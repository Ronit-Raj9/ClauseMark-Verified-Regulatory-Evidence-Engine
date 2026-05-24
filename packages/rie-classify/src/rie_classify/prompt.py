"""Pure prompt-builder for the constrained-decoding classifier.

No I/O, no LLM calls — just deterministic string formatting. Tested in
isolation. The prompt enumerates the candidate indicators, the clause text,
and the neighbourhood elements, then instructs the model to (a) pick exactly
one indicator id from the enum, (b) emit a clause pattern, (c) decompose,
and (d) cite by element id — NEVER by writing a citation string.
"""

from __future__ import annotations

from collections.abc import Sequence

from rie_contracts import Element, IndicatorConfig, PillarConfig

SYSTEM_INSTRUCTION = (
    "You are a regulatory-classification subsystem. You read one legal clause "
    "and its structural neighbourhood, then produce a single JSON object that "
    "validates against the supplied JSON schema.\n"
    "\n"
    "Hard rules you cannot break (the decoder enforces these):\n"
    "1. `indicator_id` MUST be exactly one of the enumerated indicator ids — "
    "the grammar will mask every other token.\n"
    "2. `clause_pattern` MUST be one of: obligation, prohibition, "
    "conditional_regime, exemption, definition.\n"
    "3. `evidence_element_ids` and `regime_member_ids` are lists of element "
    "ids drawn from the clause + neighbourhood. NEVER write a citation "
    "string, page number, or quoted text — only element ids.\n"
    "4. Decompose the clause into subject / condition / constraint / context "
    "(Compliance-to-Code style).\n"
    "5. If no indicator fits, pick the closest indicator id from the enum "
    "and reflect uncertainty in your decomposition — you may not abstain."
)


def _format_indicator(ic: IndicatorConfig) -> str:
    lines: list[str] = [
        f"- id: {ic.indicator_id}",
        f"  name: {ic.name}",
        f"  definition: {ic.definition.strip()}",
        f"  evaluation_method: {ic.evaluation_method.value}",
    ]
    if ic.clause_pattern is not None:
        lines.append(f"  typical_clause_pattern: {ic.clause_pattern.value}")
    if ic.scoring_criteria:
        crit = "; ".join(f"{band}={desc}" for band, desc in ic.scoring_criteria.items())
        lines.append(f"  scoring_criteria: {crit}")
    if ic.positive_keywords:
        for lang, kws in ic.positive_keywords.items():
            lines.append(f"  positive_keywords[{lang}]: {', '.join(kws)}")
    if ic.negative_cues:
        for lang, kws in ic.negative_cues.items():
            lines.append(f"  negative_cues[{lang}]: {', '.join(kws)}")
    if ic.few_shot_examples:
        lines.append("  few_shot_examples:")
        for ex in ic.few_shot_examples:
            text = str(ex.get("text", "")).strip().replace("\n", " ")
            label = ex.get("label", "")
            rationale = str(ex.get("rationale", "")).strip().replace("\n", " ")
            lines.append(f"    * text: {text}")
            lines.append(f"      label: {label}")
            if rationale:
                lines.append(f"      rationale: {rationale}")
    return "\n".join(lines)


def _format_neighbourhood(neighbourhood: Sequence[Element]) -> str:
    if not neighbourhood:
        return "(no neighbourhood elements provided)"
    chunks: list[str] = []
    for e in neighbourhood:
        chunks.append(
            "\n".join(
                [
                    f"- element_id: {e.element_id}",
                    f"  type: {e.element_type.value}",
                    f"  legal_numbering: {e.legal_numbering or '(none)'}",
                    f"  char_offsets: {e.char_start}-{e.char_end}",
                    f"  text: {e.text.strip()}",
                ]
            )
        )
    return "\n".join(chunks)


def build_classification_prompt(
    clause_element: Element,
    neighbourhood: Sequence[Element],
    pillar: PillarConfig,
    indicator_choices: Sequence[IndicatorConfig],
) -> str:
    """Assemble the full prompt — system instruction + pillar + indicators + clause."""

    indicators_block = "\n\n".join(_format_indicator(ic) for ic in indicator_choices)
    allowed_ids = ", ".join(ic.indicator_id for ic in indicator_choices)
    neighbourhood_block = _format_neighbourhood(neighbourhood)

    return (
        f"{SYSTEM_INSTRUCTION}\n"
        "\n"
        f"## Pillar\n"
        f"pillar_id: {pillar.pillar_id}\n"
        f"pillar_name: {pillar.pillar_name}\n"
        f"cluster: {pillar.cluster}\n"
        f"description: {pillar.description.strip()}\n"
        "\n"
        f"## Candidate indicators (the ONLY allowed indicator_id values: {allowed_ids})\n"
        f"{indicators_block}\n"
        "\n"
        f"## Primary clause\n"
        f"element_id: {clause_element.element_id}\n"
        f"doc_id: {clause_element.doc_id}\n"
        f"legal_numbering: {clause_element.legal_numbering or '(none)'}\n"
        f"char_offsets: {clause_element.char_start}-{clause_element.char_end}\n"
        f"text: {clause_element.text.strip()}\n"
        "\n"
        f"## Neighbourhood elements (use these element_ids in your citations)\n"
        f"{neighbourhood_block}\n"
        "\n"
        "## Task\n"
        "Return a JSON object that validates against the supplied schema. "
        "Cite by element_id only. Do not invent ids."
    )


__all__ = ["SYSTEM_INSTRUCTION", "build_classification_prompt"]
