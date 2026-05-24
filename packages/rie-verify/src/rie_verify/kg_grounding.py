"""KG / entity-grounding gate (Phase 2 — opt-in fifth verifier gate).

This module implements a deterministic, lightweight legal-text entity
extractor and a per-document knowledge graph used by a *fifth* verifier
gate ("kg_grounding") that runs alongside the canonical 4 gates.

Honest scope — read this before relying on the gate
---------------------------------------------------
The entity extractor here is a **heuristic, pure-Python NER** based on
regular expressions tuned for common patterns in statutory English:

* defined-terms found via the construct
  ``"<term>" (shall mean|means|refers to) ...``
* cross-references like ``section 26(2)`` / ``Article 4`` / ``paragraph 1(a)``
* ISO 8601 dates and common legal date forms (``1 January 2020``, ``2020-01-01``)
* monetary amounts (``USD 1,000,000`` / ``$50``)

It is **not** a legal-tuned NER. It will miss entities that do not match
these patterns, and it will produce false positives on prose that
coincidentally matches them. We rely on the orchestration policy that
*any* KG-resolution failure → ``flagged`` (never auto-rejected), so a
heuristic miss surfaces a claim for human review rather than dropping it
silently. See ``CLAUDE.md`` (§ "Verification gates") and the project
roadmap for the planned upgrade to a model-backed NER.

Contract-enum note
------------------
The frozen ``rie_contracts.GateName`` enum has 4 values (Phase 1). The
fifth gate is therefore identified by the string literal ``"kg_grounding"``
in this module and in the returned ``KgGroundingResult``. A future
contracts revision should add ``GateName.KG_GROUNDING``; until then we
keep the new gate result on a separate channel (``VerificationReport``
only accepts the original 4 ``GateResult`` entries — its model_validator
rejects extras).
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Final

from rie_contracts import Claim
from rie_contracts.ports import ElementTextResolver

KG_GATE_NAME: Final[str] = "kg_grounding"


class EntityKind(StrEnum):
    DEFINED_TERM = "defined_term"
    CROSS_REFERENCE = "cross_reference"
    DATE = "date"
    MONEY = "money"


@dataclass(frozen=True)
class Entity:
    """A normalised entity surface form + its kind. ``key`` is used for KG joins."""

    kind: EntityKind
    surface: str
    key: str  # case-/whitespace-normalised join key

    @staticmethod
    def make(kind: EntityKind, surface: str) -> Entity:
        return Entity(kind=kind, surface=surface, key=_normalise(surface))


@dataclass
class DocumentKg:
    """Per-document KG.

    ``nodes`` is a kind->set-of-keys index.
    ``edges`` is an undirected adjacency map keyed by entity key (we co-locate
    entity keys that appeared in the same element, plus structure-graph parent
    co-location). The edges are not consulted by the gate at the moment — the
    gate only needs node lookups — but they are exposed for future extensions
    (e.g. relation-grounded checks).
    """

    nodes: dict[EntityKind, set[str]] = field(
        default_factory=lambda: defaultdict(set),  # type: ignore[arg-type]
    )
    edges: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))  # type: ignore[arg-type]

    def add_entity(self, entity: Entity) -> None:
        self.nodes[entity.kind].add(entity.key)

    def add_cooccurrence(self, entities: Sequence[Entity]) -> None:
        keys = [e.key for e in entities]
        for i, a in enumerate(keys):
            for b in keys[i + 1 :]:
                if a == b:
                    continue
                self.edges[a].add(b)
                self.edges[b].add(a)

    def has(self, key: str) -> bool:
        norm = _normalise(key)
        return any(norm in keys for keys in self.nodes.values())

    def all_keys(self) -> set[str]:
        out: set[str] = set()
        for keys in self.nodes.values():
            out |= keys
        return out


@dataclass(frozen=True)
class KgGroundingResult:
    """Outcome of the KG-grounding gate.

    Two outcomes only — pass or flag. Per CLAUDE.md ("any disagreement →
    flagged, never auto-resolved"), an unresolved entity does NOT reject the
    claim; it surfaces it for human review.
    """

    gate: str
    passed: bool
    detail: str
    flagged: bool
    unresolved: tuple[str, ...]
    ran_at: datetime


# ──────────────────────────────────────────────────────────────────────────
# Regex patterns
# ──────────────────────────────────────────────────────────────────────────

_DEFINED_TERM_PATTERN: Final[re.Pattern[str]] = re.compile(
    r'"([^"\n]{1,120})"\s+(?:shall\s+mean|means|refers\s+to)\b',
    re.IGNORECASE,
)

_CROSS_REF_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b(?:section|article|paragraph|sub-section|sub-paragraph)\s+\d+[A-Za-z]?"
    r"(?:\(\d+[A-Za-z]?\))?",
    re.IGNORECASE,
)

_ISO_DATE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])\b"
)

_LEGAL_DATE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b(?:\d{1,2}(?:st|nd|rd|th)?\s+)?"
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"(?:\s+\d{1,2}(?:st|nd|rd|th)?,?)?\s+\d{4}\b",
    re.IGNORECASE,
)

_MONEY_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?:"
    r"(?:USD|EUR|GBP|JPY|INR|SGD|AUD|CAD|CHF|CNY)\s?\d[\d,]*(?:\.\d+)?"
    r"|"
    r"[$£€¥₹]\s?\d[\d,]*(?:\.\d+)?"
    r")"
)


def _normalise(text: str) -> str:
    """Lowercase, collapse internal whitespace, strip — the KG join key."""
    return " ".join(text.lower().split())


# ──────────────────────────────────────────────────────────────────────────
# Entity extractor
# ──────────────────────────────────────────────────────────────────────────


def extract_entities(text: str) -> list[Entity]:
    """Run all four heuristic NER passes over ``text``.

    Order: defined-terms first (so we capture the quoted lemma before the
    cross-reference pass), then cross-refs, dates, money. Duplicates within
    a single call are returned once (de-duped on ``(kind, key)``).
    """
    found: list[Entity] = []
    seen: set[tuple[EntityKind, str]] = set()

    def _emit(kind: EntityKind, surface: str) -> None:
        ent = Entity.make(kind, surface)
        signature = (ent.kind, ent.key)
        if signature in seen:
            return
        seen.add(signature)
        found.append(ent)

    for m in _DEFINED_TERM_PATTERN.finditer(text):
        _emit(EntityKind.DEFINED_TERM, m.group(1))

    for m in _CROSS_REF_PATTERN.finditer(text):
        _emit(EntityKind.CROSS_REFERENCE, m.group(0))

    for m in _ISO_DATE_PATTERN.finditer(text):
        _emit(EntityKind.DATE, m.group(0))

    for m in _LEGAL_DATE_PATTERN.finditer(text):
        _emit(EntityKind.DATE, m.group(0))

    for m in _MONEY_PATTERN.finditer(text):
        _emit(EntityKind.MONEY, m.group(0))

    return found


# ──────────────────────────────────────────────────────────────────────────
# KG construction
# ──────────────────────────────────────────────────────────────────────────


def build_document_kg(
    element_texts: Mapping[str, str],
    parent_of: Mapping[str, str | None] | None = None,
) -> DocumentKg:
    """Build a tiny per-document KG from ``element_id -> element_text``.

    Co-occurrence within an element produces edges. If a ``parent_of`` map
    is supplied, entities of an element are additionally co-located with
    entities of the element's structural parent (one hop only).
    """
    kg = DocumentKg()
    per_element: dict[str, list[Entity]] = {}

    for element_id, text in element_texts.items():
        entities = extract_entities(text)
        per_element[element_id] = entities
        for ent in entities:
            kg.add_entity(ent)
        kg.add_cooccurrence(entities)

    if parent_of:
        for element_id, parent in parent_of.items():
            if parent is None:
                continue
            child_ents = per_element.get(element_id, [])
            parent_ents = per_element.get(parent, [])
            if child_ents and parent_ents:
                kg.add_cooccurrence([*child_ents, *parent_ents])

    return kg


# ──────────────────────────────────────────────────────────────────────────
# Gate
# ──────────────────────────────────────────────────────────────────────────


def _claim_referenced_entities(claim: Claim) -> list[Entity]:
    """Pull entities out of the claim's decomposition.subject + constraint."""
    d = claim.decomposition
    parts: list[str] = [d.subject, d.constraint]
    if d.condition:
        parts.append(d.condition)
    if d.context:
        parts.append(d.context)
    blob = " . ".join(parts)
    return extract_entities(blob)


def run_kg_grounding_gate(
    claim: Claim,
    get_element_text: ElementTextResolver,
    *,
    parent_of: Mapping[str, str | None] | None = None,
    extra_element_ids: Iterable[str] = (),
) -> KgGroundingResult:
    """Run the KG-grounding gate for a single claim.

    Builds a per-document KG over every element id referenced by the
    claim's evidence_spans (plus any ``extra_element_ids`` the caller wants
    to include — typically regime members). Then extracts entities from the
    claim's decomposition and checks each one resolves in the KG.

    Pass ⇔ every claim-side entity key resolves.
    Flag ⇔ ≥ 1 unresolved key (never auto-reject).
    """
    now = datetime.now(UTC)

    element_ids: list[str] = []
    seen_ids: set[str] = set()
    for span in claim.evidence_spans:
        if span.element_id not in seen_ids:
            seen_ids.add(span.element_id)
            element_ids.append(span.element_id)
    for eid in extra_element_ids:
        if eid not in seen_ids:
            seen_ids.add(eid)
            element_ids.append(eid)

    element_texts: dict[str, str] = {}
    resolve_problems: list[str] = []
    for eid in element_ids:
        try:
            element_texts[eid] = get_element_text(eid)
        except (KeyError, LookupError, ValueError) as exc:
            resolve_problems.append(f"{eid}: {exc}")

    if not element_texts:
        return KgGroundingResult(
            gate=KG_GATE_NAME,
            passed=False,
            flagged=True,
            unresolved=(),
            detail=(
                "kg_grounding flagged: no element text resolvable for claim "
                f"({'; '.join(resolve_problems) or 'no evidence_spans'})"
            ),
            ran_at=now,
        )

    kg = build_document_kg(element_texts, parent_of=parent_of)

    claim_entities = _claim_referenced_entities(claim)
    if not claim_entities:
        # Nothing named in the decomposition — vacuously grounded. We do NOT
        # flag in this case because the heuristic extractor not firing on
        # the decomposition tells us nothing about the underlying claim.
        return KgGroundingResult(
            gate=KG_GATE_NAME,
            passed=True,
            flagged=False,
            unresolved=(),
            detail="kg_grounding pass: no named entities in claim decomposition (vacuous)",
            ran_at=now,
        )

    unresolved: list[str] = []
    for ent in claim_entities:
        if not kg.has(ent.key):
            unresolved.append(f"{ent.kind.value}:{ent.surface}")

    if unresolved:
        return KgGroundingResult(
            gate=KG_GATE_NAME,
            passed=False,
            flagged=True,
            unresolved=tuple(unresolved),
            detail=(
                f"kg_grounding flagged: {len(unresolved)} unresolved "
                f"entit{'y' if len(unresolved) == 1 else 'ies'} "
                f"({', '.join(unresolved)})"
            ),
            ran_at=now,
        )

    return KgGroundingResult(
        gate=KG_GATE_NAME,
        passed=True,
        flagged=False,
        unresolved=(),
        detail=f"kg_grounding pass: all {len(claim_entities)} entities resolve in document KG",
        ran_at=now,
    )


__all__ = [
    "KG_GATE_NAME",
    "DocumentKg",
    "Entity",
    "EntityKind",
    "KgGroundingResult",
    "build_document_kg",
    "extract_entities",
    "run_kg_grounding_gate",
]
