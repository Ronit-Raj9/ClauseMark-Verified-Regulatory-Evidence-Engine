"""Tests for the Discovery NEW/KNOWN tagger.

Covers the two matching rules (doc+indicator, fuzzy span overlap) plus the
negative cases (unknown doc, wrong jurisdiction) and indicator normalisation.
Pure functions — no I/O, no network, no LLM.
"""

from __future__ import annotations

import pytest
from rie_contracts import AuthorityTier, ClausePattern, GoldItem, ScoreBand
from rie_ingest.discovery_tag import (
    TAG_KNOWN,
    TAG_NEW,
    DiscoveryRecord,
    DiscoveryTagger,
    load_gold_for,
    normalize_indicator_id,
    tag_discovery,
    token_overlap,
)

_SG_64_SPAN = (
    "An organisation shall not transfer any personal data outside Singapore "
    "unless the organisation has taken appropriate steps to ensure that the "
    "recipient is bound by legally enforceable obligations to provide a "
    "standard of protection comparable to the protection under this Act."
)


def _gold(
    *,
    gold_id: str,
    indicator_id: str,
    jurisdiction: str,
    doc_id: str,
    span_text: str,
    pillar_id: str = "6",
) -> GoldItem:
    return GoldItem(
        gold_id=gold_id,
        pillar_id=pillar_id,
        indicator_id=indicator_id,
        jurisdiction=jurisdiction,
        doc_id=doc_id,
        span_text=span_text,
        expected_clause_pattern=ClausePattern.CONDITIONAL_REGIME,
        expected_score_band=ScoreBand.ONE,
        expected_authority_tier=AuthorityTier.TIER_1_STATUTE,
    )


@pytest.fixture
def gold() -> list[GoldItem]:
    return [
        _gold(
            gold_id="sg_64",
            indicator_id="6.4",
            jurisdiction="Singapore",
            doc_id="sg_pdpa_2012",
            span_text=_SG_64_SPAN,
        ),
        _gold(
            gold_id="my_61",
            indicator_id="6.1",
            jurisdiction="Malaysia",
            doc_id="my_pdpa_act709_2010",
            span_text=(
                "A data user shall not transfer any personal data to a place "
                "outside Malaysia unless that place has been specified by the "
                "Minister."
            ),
        ),
    ]


# ─── KNOWN: exact gold match ─────────────────────────────────────────────────


def test_exact_gold_match_is_known(gold: list[GoldItem]) -> None:
    tag = tag_discovery(
        jurisdiction="Singapore",
        indicator_id="6.4",
        doc_id="sg_pdpa_2012",
        span_text=_SG_64_SPAN,
        gold=gold,
    )
    assert tag == TAG_KNOWN


# ─── KNOWN: same doc + indicator, different span (doc+indicator rule) ─────────


def test_same_doc_and_indicator_different_span_is_known(gold: list[GoldItem]) -> None:
    tag = tag_discovery(
        jurisdiction="Singapore",
        indicator_id="6.4",
        doc_id="sg_pdpa_2012",
        # Wholly different text — but same law, same indicator ⇒ KNOWN.
        span_text="Totally unrelated wording about widgets and sprockets.",
        gold=gold,
    )
    assert tag == TAG_KNOWN


# ─── NEW: unknown doc (and no fuzzy overlap) ─────────────────────────────────


def test_unknown_doc_is_new(gold: list[GoldItem]) -> None:
    tag = tag_discovery(
        jurisdiction="Singapore",
        indicator_id="6.4",
        doc_id="sg_some_other_act_2099",
        span_text="A bespoke clause with no overlap to the gold provision text.",
        gold=gold,
    )
    assert tag == TAG_NEW


# ─── KNOWN: fuzzy whitespace / case match on a different doc id ───────────────


def test_fuzzy_whitespace_and_case_is_known(gold: list[GoldItem]) -> None:
    # Different doc_id (so the doc+indicator rule does NOT fire), but the span
    # is the gold span re-cased + re-whitespaced ⇒ fuzzy rule fires.
    noisy_span = "  AN ORGANISATION   shall NOT transfer\tany PERSONAL data\n outside Singapore "
    noisy_span += (
        "unless the organisation has taken appropriate steps to ensure that the recipient is "
        "bound by legally enforceable obligations to provide a standard of protection comparable "
        "to the protection under this Act."
    )
    tag = tag_discovery(
        jurisdiction="Singapore",
        indicator_id="6.4",
        doc_id="sg_pdpa_mirror_copy",  # different id → only fuzzy can match
        span_text=noisy_span,
        gold=gold,
    )
    assert tag == TAG_KNOWN


# ─── NEW: wrong jurisdiction ─────────────────────────────────────────────────


def test_wrong_jurisdiction_is_new(gold: list[GoldItem]) -> None:
    # Identical doc_id + indicator + span as the Singapore gold row, but a
    # different jurisdiction ⇒ NEW (jurisdiction is part of the match key).
    tag = tag_discovery(
        jurisdiction="Australia",
        indicator_id="6.4",
        doc_id="sg_pdpa_2012",
        span_text=_SG_64_SPAN,
        gold=gold,
    )
    assert tag == TAG_NEW


# ─── NEW: right doc, wrong indicator, low overlap ────────────────────────────


def test_wrong_indicator_is_new(gold: list[GoldItem]) -> None:
    tag = tag_discovery(
        jurisdiction="Singapore",
        indicator_id="7.1",  # gold has 6.4 for this doc, not 7.1
        doc_id="sg_pdpa_2012",
        span_text="A completely different storage-period clause.",
        gold=gold,
    )
    assert tag == TAG_NEW


# ─── Indicator normalisation ─────────────────────────────────────────────────


def test_indicator_alias_form_is_known(gold: list[GoldItem]) -> None:
    # "P6-I4" display alias normalises to "6.4".
    tag = tag_discovery(
        jurisdiction="Singapore",
        indicator_id="P6-I4",
        doc_id="sg_pdpa_2012",
        span_text="anything",
        gold=gold,
    )
    assert tag == TAG_KNOWN


def test_normalize_indicator_id_variants() -> None:
    assert normalize_indicator_id("6.4") == "6.4"
    assert normalize_indicator_id(" 6.4 ") == "6.4"
    assert normalize_indicator_id("P6-I4") == "6.4"
    assert normalize_indicator_id("p7-i5") == "7.5"
    # Trailing zeros are NEVER collapsed — 4.1 and 4.10 are distinct decimals,
    # so the normaliser leaves the fractional part exactly as authored.
    assert normalize_indicator_id("4.10") == "4.10"
    assert normalize_indicator_id("6.40") == "6.40"


def test_empty_gold_is_new() -> None:
    assert (
        tag_discovery(
            jurisdiction="Singapore",
            indicator_id="6.4",
            doc_id="sg_pdpa_2012",
            span_text="x",
            gold=[],
        )
        == TAG_NEW
    )


# ─── token_overlap helper ────────────────────────────────────────────────────


def test_token_overlap_identical_is_one() -> None:
    assert token_overlap("hello world", "HELLO   world") == 1.0


def test_token_overlap_disjoint_is_zero() -> None:
    assert token_overlap("alpha beta", "gamma delta") == 0.0


def test_token_overlap_empty_both_is_one() -> None:
    assert token_overlap("", "") == 1.0


# ─── DiscoveryTagger + load_gold_for ─────────────────────────────────────────


def test_tagger_tag_record(gold: list[GoldItem]) -> None:
    tagger = DiscoveryTagger(gold=gold)
    rec = DiscoveryRecord(
        jurisdiction="Malaysia",
        indicator_id="6.1",
        doc_id="my_pdpa_act709_2010",
        span_text="any wording",
    )
    assert tagger.tag(rec) == TAG_KNOWN
    assert (
        tagger.tag_fields(
            jurisdiction="Malaysia",
            indicator_id="6.1",
            doc_id="unknown_doc",
            span_text="unrelated wording about something else entirely",
        )
        == TAG_NEW
    )


def test_load_gold_for_builds_tagger(gold: list[GoldItem]) -> None:
    class _StubRepo:
        def load_gold(self, pillar_id: str) -> list[GoldItem]:
            assert pillar_id == "6"
            return gold

    tagger = load_gold_for(_StubRepo(), "6")
    assert isinstance(tagger, DiscoveryTagger)
    assert tagger.tag_fields("Singapore", "6.4", "sg_pdpa_2012", "x") == TAG_KNOWN


def test_load_gold_for_rejects_bad_repo() -> None:
    with pytest.raises(TypeError):
        load_gold_for(object(), "6")
