"""Discovery NEW/KNOWN tagger — pure, deterministic provenance classification.

The hackathon output schema (see `hackathon-output-schema.md`) requires a
`discovery_tag` column on every extracted provision:

  * ``KNOWN``  — the provision already exists in the Round-1 gold database
    (the sample kit). Judges expect these to be recognised, not re-discovered.
  * ``NEW``    — an independent find not present in the gold set.

This module decides KNOWN vs NEW *deterministically* from the gold corpus. It
performs NO LLM calls and reads nothing from the network — it is a pure function
over `(jurisdiction, indicator_id, doc_id, span_text, gold)`.

Matching rules (in order, first KNOWN wins):

  1. **doc + indicator rule** — a gold item with the same `jurisdiction`,
     `indicator_id` (canonical decimal form, pillar-dot-indicator) and `doc_id`
     ⇒ ``KNOWN``, regardless of span text. The same statute scored against the
     same indicator is the same provision family in the Round-1 DB (one
     indicator → many provisions, distinguished by *law*, not by exact span —
     see the gold data note's "GOTCHAS" #2).
  2. **fuzzy span rule** — otherwise, a gold item matching `jurisdiction` +
     `indicator_id` whose normalised span text has token-overlap above the
     threshold (default 0.6) with the candidate span ⇒ ``KNOWN`` (handles
     whitespace / case / minor edits).

Anything else ⇒ ``NEW``.

`indicator_id` is compared in canonical decimal form (pillar-dot-indicator);
a defensive normaliser tolerates surrounding whitespace and the ``P<n>-I<m>``
display alias, mapping it back to the decimal form.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from rie_contracts import GoldItem


@runtime_checkable
class _GoldSource(Protocol):
    """Structural type for anything that can hand us a gold set by pillar.

    Matches `ConfigRepositoryPort` / `rie_config.ConfigRepository` without
    importing the concrete class, and lets tests pass a stub.
    """

    def load_gold(self, pillar_id: str) -> Sequence[GoldItem]: ...


# Default Jaccard token-overlap threshold for the fuzzy span rule.
DEFAULT_TOKEN_OVERLAP_THRESHOLD: float = 0.6

# Discovery tag literals — kept as module constants so callers never typo them.
TAG_KNOWN: str = "KNOWN"
TAG_NEW: str = "NEW"

# Matches the display-alias form (P<pillar>-I<indicator>) → decimal form.
_ALIAS_RE: re.Pattern[str] = re.compile(r"^P(\d+)-I(\d+)$", re.IGNORECASE)
_WORD_RE: re.Pattern[str] = re.compile(r"\w+", re.UNICODE)


# ─── Normalisation helpers (pure) ────────────────────────────────────────────


def normalize_indicator_id(indicator_id: str) -> str:
    """Canonicalise an indicator id to canonical decimal form.

    Tolerates:
      * the ``P<pillar>-I<indicator>`` display alias, mapped back to the
        pillar-dot-indicator decimal form,
      * surrounding whitespace.

    NO trailing-zero collapsing is performed: the gold data note warns that
    multi-digit fractions collide as floats and are legitimately distinct
    (the first-vs-tenth-indicator ambiguity). Stripping a trailing zero would
    conflate them, so we leave the fractional part exactly as authored. The
    gold YAML and the engine must agree on the same string form — that is the
    contract, and this normaliser keeps it byte-stable.
    """
    raw = indicator_id.strip()
    alias = _ALIAS_RE.match(raw)
    if alias:
        return f"{int(alias.group(1))}.{int(alias.group(2))}"
    return raw


def _tokens(text: str) -> frozenset[str]:
    """Lower-cased word-token set of `text` (whitespace/case insensitive)."""
    return frozenset(m.group(0) for m in _WORD_RE.finditer(text.lower()))


def token_overlap(a: str, b: str) -> float:
    """Jaccard similarity over word-token sets. Empty-vs-empty ⇒ 1.0."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


# ─── Core tag function (pure) ────────────────────────────────────────────────


def tag_discovery(
    jurisdiction: str,
    indicator_id: str,
    doc_id: str,
    span_text: str,
    gold: Sequence[GoldItem],
    *,
    token_overlap_threshold: float = DEFAULT_TOKEN_OVERLAP_THRESHOLD,
) -> str:
    """Return ``"KNOWN"`` if a gold item matches this provision, else ``"NEW"``.

    See module docstring for the two-rule matching contract. Pure +
    deterministic: identical inputs always produce identical output.
    """
    jur = jurisdiction.strip()
    ind = normalize_indicator_id(indicator_id)
    doc = doc_id.strip()

    # Pre-filter gold to the (jurisdiction, indicator) cell once.
    cell: list[GoldItem] = [
        g
        for g in gold
        if g.jurisdiction.strip() == jur and normalize_indicator_id(g.indicator_id) == ind
    ]
    if not cell:
        return TAG_NEW

    # Rule 1 — doc + indicator alone is KNOWN (same law, same indicator).
    for g in cell:
        if g.doc_id.strip() == doc:
            return TAG_KNOWN

    # Rule 2 — fuzzy span overlap within the (jurisdiction, indicator) cell.
    for g in cell:
        if token_overlap(span_text, g.span_text) >= token_overlap_threshold:
            return TAG_KNOWN

    return TAG_NEW


# ─── DiscoveryTagger — bound to a gold corpus ────────────────────────────────


@dataclass(frozen=True)
class DiscoveryRecord:
    """The minimal addressable shape a tagger needs to classify a provision.

    Decoupled from `Claim`/`Element` so the tagger stays in `rie-ingest` (which
    may not import the classify/extract packages) and remains trivially
    constructable in tests and from CSV rows.
    """

    jurisdiction: str
    indicator_id: str
    doc_id: str
    span_text: str


@dataclass(frozen=True)
class DiscoveryTagger:
    """Stateful wrapper holding a gold corpus, with a `tag(record)` method.

    Construct once per run (gold is loaded once), then classify many records.
    """

    gold: Sequence[GoldItem]
    token_overlap_threshold: float = DEFAULT_TOKEN_OVERLAP_THRESHOLD

    def tag(self, record: DiscoveryRecord) -> str:
        """Tag a single `DiscoveryRecord` as ``"KNOWN"`` or ``"NEW"``."""
        return tag_discovery(
            jurisdiction=record.jurisdiction,
            indicator_id=record.indicator_id,
            doc_id=record.doc_id,
            span_text=record.span_text,
            gold=self.gold,
            token_overlap_threshold=self.token_overlap_threshold,
        )

    def tag_fields(self, jurisdiction: str, indicator_id: str, doc_id: str, span_text: str) -> str:
        """Field-wise convenience wrapper around `tag`."""
        return self.tag(
            DiscoveryRecord(
                jurisdiction=jurisdiction,
                indicator_id=indicator_id,
                doc_id=doc_id,
                span_text=span_text,
            )
        )


def load_gold_for(config_repo: _GoldSource, pillar_id: str) -> DiscoveryTagger:
    """Build a `DiscoveryTagger` from a config repository's gold set.

    `config_repo` is any object satisfying `_GoldSource` (i.e. exposing
    `load_gold(pillar_id) -> Sequence[GoldItem]`), which the
    `ConfigRepositoryPort` / `rie_config.ConfigRepository` do. The structural
    `Protocol` keeps `rie-ingest` from importing the concrete config class while
    still letting tests inject a stub. A non-conforming object (passed despite
    the type hint) raises `TypeError` rather than an opaque `AttributeError`.
    """
    if not hasattr(config_repo, "load_gold"):
        raise TypeError(
            "config_repo must expose a callable `load_gold(pillar_id)` "
            f"(got {type(config_repo).__name__})"
        )
    return DiscoveryTagger(gold=config_repo.load_gold(pillar_id))
