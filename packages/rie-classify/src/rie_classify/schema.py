"""Constrained-decoding output schema.

The classifier LLM is *structurally unable* to fabricate an `indicator_id` —
the JSON schema's ``indicator_id`` field is a ``Literal[...]`` enumeration of
the candidate ids passed into the call. Grammar-constrained decoders
(XGrammar in vLLM/Ollama, Outlines in transformers) enforce this at the token
level: the model cannot emit any token that would yield a string outside the
enum.

`build_output_model` constructs a fresh Pydantic v2 model per call so the
enum is the exact set of candidate indicator ids for *this* clause. The
resulting model's JSON schema (``model.model_json_schema()``) is what we hand
to the LLM client as the ``format`` / ``guided_json`` payload.
"""

from __future__ import annotations

from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, create_model
from rie_contracts import ClausePattern, Decomposition, IndicatorConfig


class ClassificationOutputBase(BaseModel):
    """Base shape — the dynamic subclass overrides `indicator_id` with a Literal."""

    model_config = ConfigDict(extra="forbid")

    indicator_id: str = Field(
        description=(
            "The single best-matching indicator id from the enumerated choices. "
            "MUST be exactly one of the allowed values — the grammar will reject "
            "anything else."
        )
    )
    clause_pattern: ClausePattern = Field(
        description=("One of: obligation, prohibition, conditional_regime, exemption, definition.")
    )
    decomposition: Decomposition = Field(
        description="Compliance-to-Code decomposition: subject / condition / constraint / context."
    )
    evidence_element_ids: list[str] = Field(
        min_length=1,
        description=(
            "Element ids drawn from the clause + its neighbourhood. NEVER a citation "
            "string — the materialiser converts these ids into EvidenceSpans."
        ),
    )
    regime_member_ids: list[str] = Field(
        min_length=1,
        description=(
            "Element ids forming the legal regime around the primary clause "
            "(definitions, exceptions, cross-referenced provisions)."
        ),
    )


def build_output_model(
    indicator_choices: list[IndicatorConfig] | tuple[IndicatorConfig, ...],
) -> type[ClassificationOutputBase]:
    """Return a dynamically-built Pydantic model whose `indicator_id` is a `Literal`.

    The Literal is constructed from the indicator ids in ``indicator_choices``.
    A model whose ``indicator_id`` value is not in the enum fails validation
    with a Pydantic ``ValidationError`` — and any constrained-decoding backend
    handed ``model.model_json_schema()`` will physically block emission of any
    other value.
    """

    if not indicator_choices:
        raise ValueError("indicator_choices must be non-empty")

    ids = tuple(ic.indicator_id for ic in indicator_choices)
    # Defensive: dedupe while preserving order.
    seen: set[str] = set()
    unique_ids: list[str] = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            unique_ids.append(i)

    # `Literal[*tuple]` is not subscript-valid in 3.12, so go through __class_getitem__.
    literal_type = cast(type, Literal[tuple(unique_ids)])  # type: ignore[valid-type]

    model = create_model(
        "ClassificationOutput",
        __base__=ClassificationOutputBase,
        indicator_id=(
            literal_type,
            Field(
                description=("Pick exactly one. Allowed values: " + ", ".join(unique_ids)),
            ),
        ),
    )
    return cast(type[ClassificationOutputBase], model)


__all__ = ["ClassificationOutputBase", "build_output_model"]
