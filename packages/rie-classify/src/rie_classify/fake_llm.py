"""Deterministic stand-in LLM for tests and offline development.

Picks an indicator from the candidate enum using simple keyword heuristics
over the prompt text — never reaches outside the supplied schema. Returns
JSON dicts that validate against the dynamic ``ClassificationOutput`` model.

The fake intentionally walks the same code path as a real LLM (prompt in,
JSON dict out) so the service does not branch on "is this a fake".
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from .llm_client import LlmClient


def _allowed_indicator_ids(schema: dict[str, Any]) -> list[str]:
    """Pull the Literal enum out of the schema so the fake stays in-bounds."""

    props = schema.get("properties", {})
    indicator = props.get("indicator_id", {})
    enum_values = indicator.get("enum")
    if not isinstance(enum_values, list) or not enum_values:
        raise ValueError("schema is missing an indicator_id enum")
    return [str(v) for v in enum_values]


def _extract_element_ids(prompt: str) -> list[str]:
    """Find every `element_id: ...` mention in the prompt and return them in order."""

    out: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r"element_id:\s*([\w.\-]+)", prompt):
        eid = match.group(1)
        if eid not in seen:
            seen.add(eid)
            out.append(eid)
    return out


def _primary_element_id(prompt: str) -> str | None:
    """The first element_id under the `## Primary clause` heading."""

    m = re.search(
        r"##\s*Primary clause\s*\n[\s\S]*?element_id:\s*([\w.\-]+)",
        prompt,
    )
    return m.group(1) if m else None


def _clause_text(prompt: str) -> str:
    m = re.search(r"##\s*Primary clause[\s\S]*?\ntext:\s*(.+?)(?:\n##|\Z)", prompt, re.S)
    return (m.group(1) if m else prompt).lower()


def _pick_indicator(
    clause_lower: str,
    allowed: Sequence[str],
    keyword_to_id: dict[tuple[str, ...], str],
) -> str:
    """Apply keyword heuristics then fall back to the first allowed id."""

    for keywords, indicator_id in keyword_to_id.items():
        if indicator_id not in allowed:
            continue
        if all(kw in clause_lower for kw in keywords):
            return indicator_id
    return allowed[0]


class DeterministicFakeLlm(LlmClient):
    """A deterministic, offline LLM substitute. Same wire shape as real clients.

    Heuristics are pure pattern matches over the rendered prompt. Tests pass
    in a ``keyword_to_indicator`` mapping if they want bespoke behaviour;
    otherwise the built-in defaults cover the cross-border conditional-
    regime case used in the canonical end-to-end test.
    """

    def __init__(
        self,
        keyword_to_indicator: dict[tuple[str, ...], str] | None = None,
        force_indicator: str | None = None,
        force_pattern: str | None = None,
    ) -> None:
        # Default heuristics — abstract, not pillar-specific. Tests inject
        # the concrete indicator id they care about via
        # ``keyword_to_indicator`` so the engine code never hard-codes one.
        self.keyword_to_indicator: dict[tuple[str, ...], str] = (
            keyword_to_indicator if keyword_to_indicator is not None else {}
        )
        self.force_indicator = force_indicator
        self.force_pattern = force_pattern

    def complete_json(
        self,
        prompt: str,
        schema: dict[str, Any],
        temperature: float,
        seed: int | None,
    ) -> dict[str, Any]:
        allowed = _allowed_indicator_ids(schema)
        clause = _clause_text(prompt)
        element_ids = _extract_element_ids(prompt)
        primary = _primary_element_id(prompt) or (element_ids[0] if element_ids else "unknown")

        if self.force_indicator is not None and self.force_indicator in allowed:
            indicator_id = self.force_indicator
        else:
            indicator_id = _pick_indicator(clause, allowed, self.keyword_to_indicator)

        if self.force_pattern is not None:
            pattern = self.force_pattern
        elif "shall not" in clause and (
            "unless" in clause or "provided that" in clause or "comparable" in clause
        ):
            pattern = "conditional_regime"
        elif "shall not" in clause or "prohibited" in clause:
            pattern = "prohibition"
        elif "means" in clause and ("personal data" in clause or "definition" in clause):
            pattern = "definition"
        elif "exempt" in clause:
            pattern = "exemption"
        else:
            pattern = "obligation"

        # Slightly vary decomposition via temperature/seed so self-consistency
        # tests can simulate disagreement when a caller wants it. Pure
        # determinism on a fixed (prompt, temperature, seed) tuple is
        # preserved.
        constraint = "constraint extracted from clause"
        if seed is not None and seed % 2 == 1 and temperature > 0.4:
            constraint = "constraint extracted (alt sampling)"

        return {
            "indicator_id": indicator_id,
            "clause_pattern": pattern,
            "decomposition": {
                "subject": "regulated entity",
                "condition": "as specified in clause",
                "constraint": constraint,
                "context": None,
            },
            "evidence_element_ids": [primary] + [e for e in element_ids if e != primary][:2],
            "regime_member_ids": [primary] + [e for e in element_ids if e != primary],
        }


class ScriptedFakeLlm(LlmClient):
    """Cycle through a fixed list of pre-baked JSON outputs.

    Useful for self-consistency tests where you need to control exactly how
    each of N samples votes.
    """

    def __init__(self, outputs: Sequence[dict[str, Any]]) -> None:
        if not outputs:
            raise ValueError("outputs must be non-empty")
        self._outputs = list(outputs)
        self._idx = 0

    def complete_json(
        self,
        prompt: str,
        schema: dict[str, Any],
        temperature: float,
        seed: int | None,
    ) -> dict[str, Any]:
        out = self._outputs[self._idx % len(self._outputs)]
        self._idx += 1
        return dict(out)  # defensive copy


__all__ = ["DeterministicFakeLlm", "ScriptedFakeLlm"]
