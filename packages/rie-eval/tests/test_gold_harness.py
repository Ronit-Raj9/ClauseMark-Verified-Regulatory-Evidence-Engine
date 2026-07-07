"""Unit tests for the gold harness (Substantive-Accuracy measurement).

Synthetic gold modelled on the Round-1 DB shape — crucially the
one-indicator-many-laws fact (Malaysia 6.2 = 5 provisions) so we can prove the
harness scores per (indicator, law) pair, not per indicator.
"""

from __future__ import annotations

from typing import Any

from rie_contracts import GoldItem
from rie_eval.gold_harness import (
    DISCOVERY_NEW,
    match_provisions,
    score_against_gold,
)

# ─── Fixtures ────────────────────────────────────────────────────────────────


def _gold(
    gold_id: str,
    indicator_id: str,
    doc_id: str,
    *,
    jurisdiction: str = "Malaysia",
    pillar_id: str = "6",
    span_text: str = "An organisation shall not transfer personal data outside the country.",
    score_band: str = "1",
    authority_tier: str = "tier_1_statute",
    clause_pattern: str = "prohibition",
) -> GoldItem:
    return GoldItem.model_validate(
        {
            "gold_id": gold_id,
            "pillar_id": pillar_id,
            "indicator_id": indicator_id,
            "jurisdiction": jurisdiction,
            "doc_id": doc_id,
            "span_text": span_text,
            "expected_clause_pattern": clause_pattern,
            "expected_score_band": score_band,
            "expected_authority_tier": authority_tier,
        }
    )


def _pred(
    indicator_id: str,
    doc_id: str,
    *,
    jurisdiction: str = "Malaysia",
    span_text: str = "An organisation shall not transfer personal data outside the country.",
    score_band: str = "1",
    authority_tier: str = "tier_1_statute",
    discovery_tag: str = "KNOWN",
) -> dict[str, Any]:
    return {
        "jurisdiction": jurisdiction,
        "indicator_id": indicator_id,
        "doc_id": doc_id,
        "span_text": span_text,
        "score_band": score_band,
        "authority_tier": authority_tier,
        "discovery_tag": discovery_tag,
    }


# Malaysia 6.2 — five distinct provisions (laws), as in the real gold DB.
def _malaysia_62_gold() -> list[GoldItem]:
    return [
        _gold("g1", "6.2", "pdpa_2010", score_band="0"),
        _gold("g2", "6.2", "banking_code_2017", score_band="0"),
        _gold("g3", "6.2", "comms_code_2017", score_band="0"),
        _gold("g4", "6.2", "service_tax_act_2018", score_band="0.5"),
        _gold("g5", "6.2", "income_tax_act_1967", score_band="0.5"),
    ]


def _malaysia_62_pred() -> list[dict[str, Any]]:
    return [
        _pred("6.2", "pdpa_2010", score_band="0"),
        _pred("6.2", "banking_code_2017", score_band="0"),
        _pred("6.2", "comms_code_2017", score_band="0"),
        _pred("6.2", "service_tax_act_2018", score_band="0.5"),
        _pred("6.2", "income_tax_act_1967", score_band="0.5"),
    ]


# ─── Perfect prediction → f1 1.0 ─────────────────────────────────────────────


def test_perfect_prediction_scores_f1_one() -> None:
    gold = _malaysia_62_gold()
    pred = _malaysia_62_pred()
    out = score_against_gold(pred, gold)
    assert out["precision"] == 1.0
    assert out["recall"] == 1.0
    assert out["f1"] == 1.0
    assert out["field_accuracy"] == 1.0
    assert out["citation_fidelity"] == 1.0
    assert out["false_zero_rate"] == 0.0
    assert out["discovery_new_count"] == 0
    counts = out["counts"]
    assert counts["tp"] == 5
    assert counts["fn"] == 0
    assert counts["fp"] == 0


# ─── One indicator, many laws → scored per (indicator, law) pair ─────────────


def test_one_indicator_many_laws_scored_per_pair() -> None:
    """If the tool finds only ONE of the five 6.2 laws, recall must be 1/5,
    proving scoring is per-provision not per-indicator (per-indicator would
    give recall 1.0 because the indicator 6.2 'was found')."""
    gold = _malaysia_62_gold()
    pred = [_pred("6.2", "pdpa_2010", score_band="0")]  # only one of five laws
    out = score_against_gold(pred, gold)
    per = out["per_indicator"]["6.2"]
    assert per["support"] == 5.0
    assert per["recall"] == 1.0 / 5.0  # 1 of 5 provisions retrieved
    assert per["precision"] == 1.0  # the one it returned is correct
    # Macro F1 over the single indicator equals that indicator's f1.
    assert out["f1"] == per["f1"]
    counts = out["counts"]
    assert counts["tp"] == 1
    assert counts["fn"] == 4
    assert counts["fp"] == 0


def test_match_provisions_keys_by_indicator_law_triple() -> None:
    """Two different laws under the SAME indicator must NOT collapse — each is
    its own provision key and matches its own predicted row."""
    gold = [
        _gold("g1", "6.2", "service_tax_act_2018", score_band="0.5"),
        _gold("g2", "6.2", "income_tax_act_1967", score_band="0.5"),
    ]
    pred = [
        _pred("6.2", "income_tax_act_1967", score_band="0.5"),
        _pred("6.2", "service_tax_act_2018", score_band="0.5"),
    ]
    ms = match_provisions(pred, gold)
    assert len(ms.matched) == 2
    assert ms.missed == []
    assert ms.spurious == []
    # Each match pairs the correct law (doc_id) despite same indicator.
    paired_docs = {m.gold.doc_id for m in ms.matched if m.gold is not None}
    assert paired_docs == {"service_tax_act_2018", "income_tax_act_1967"}


# ─── Indicator swap → precision/recall drop ──────────────────────────────────


def test_indicator_swap_drops_precision_and_recall() -> None:
    """Predicting the wrong indicator for a found law is a wrong-indicator hit:
    the true indicator loses recall, the predicted indicator loses precision."""
    gold = [_gold("g1", "6.2", "service_tax_act_2018", score_band="0.5")]
    # Same law/jurisdiction but mislabelled indicator 6.4 instead of 6.2.
    pred = [_pred("6.4", "service_tax_act_2018", score_band="0.5")]
    out = score_against_gold(pred, gold)
    # True indicator 6.2: recall 0 (its provision matched but mislabelled).
    assert out["per_indicator"]["6.2"]["recall"] == 0.0
    # Predicted indicator 6.4: precision 0 (it claimed a 6.4 that gold says 6.2).
    assert out["per_indicator"]["6.4"]["precision"] == 0.0
    # The pair still consumes the provision (matched), so field-level shows the
    # indicator mismatch.
    assert out["indicator_accuracy"] == 0.0
    # Macro F1 strictly below the perfect case.
    assert out["f1"] < 1.0


def test_partial_indicator_swap_lowers_macro_f1() -> None:
    gold = _malaysia_62_gold()
    pred = _malaysia_62_pred()
    # Corrupt one of five predictions to a wrong indicator.
    pred[0] = _pred("6.4", "pdpa_2010", score_band="0")
    out = score_against_gold(pred, gold)
    assert 0.0 < out["f1"] < 1.0
    assert out["per_indicator"]["6.2"]["recall"] < 1.0


# ─── discovery_new_count — NEW not in gold ───────────────────────────────────


def test_discovery_new_count_counts_new_not_in_gold() -> None:
    gold = _malaysia_62_gold()
    pred = _malaysia_62_pred()
    # Add a genuinely NEW provision (a law not in the gold set) tagged NEW.
    pred.append(
        _pred(
            "6.2",
            "digital_economy_act_2099",
            span_text="A novel provision discovered beyond the sample kit.",
            discovery_tag=DISCOVERY_NEW,
        )
    )
    out = score_against_gold(pred, gold)
    assert out["discovery_new_count"] == 1
    # A NEW find is NOT counted as a spurious false positive.
    assert out["counts"]["fp"] == 0
    # The five known provisions still score perfectly.
    assert out["recall"] == 1.0
    assert out["precision"] == 1.0


def test_new_tag_on_known_provision_is_not_discovery() -> None:
    """A NEW-tagged record that actually matches gold is a normal match, not a
    discovery — the tool mislabelled a known provision."""
    gold = [_gold("g1", "6.2", "pdpa_2010", score_band="0")]
    pred = [_pred("6.2", "pdpa_2010", score_band="0", discovery_tag=DISCOVERY_NEW)]
    out = score_against_gold(pred, gold)
    assert out["discovery_new_count"] == 0
    assert out["counts"]["tp"] == 1


def test_spurious_known_prediction_counts_as_false_positive() -> None:
    """A KNOWN-tagged prediction with no gold match is a false positive."""
    gold = [_gold("g1", "6.2", "pdpa_2010", score_band="0")]
    pred = [
        _pred("6.2", "pdpa_2010", score_band="0"),
        _pred("6.2", "fabricated_law_2099", score_band="1"),  # hallucinated, KNOWN
    ]
    out = score_against_gold(pred, gold)
    assert out["counts"]["fp"] == 1
    assert out["discovery_new_count"] == 0
    assert out["per_indicator"]["6.2"]["precision"] == 0.5  # 1 of 2 correct


# ─── false_zero detected ─────────────────────────────────────────────────────


def test_false_zero_detected_when_scored_provision_missed() -> None:
    """A gold provision with score_band > 0 that the tool emits nothing for is
    a false zero (silent absence on a real restrictive provision)."""
    gold = [
        _gold("g4", "6.2", "service_tax_act_2018", score_band="0.5"),  # positive
        _gold("g5", "6.2", "income_tax_act_1967", score_band="0.5"),  # positive
        _gold("g1", "6.2", "pdpa_2010", score_band="0"),  # NOT positive
    ]
    # Tool finds only the score=0 PDPA provision; misses both 0.5 provisions.
    pred = [_pred("6.2", "pdpa_2010", score_band="0")]
    out = score_against_gold(pred, gold)
    # 2 of 2 positive provisions missed → false_zero_rate 1.0.
    assert out["false_zero_rate"] == 1.0


def test_no_false_zero_when_positive_provisions_found() -> None:
    gold = [
        _gold("g4", "6.2", "service_tax_act_2018", score_band="0.5"),
        _gold("g1", "6.2", "pdpa_2010", score_band="0"),
    ]
    pred = [
        _pred("6.2", "service_tax_act_2018", score_band="0.5"),
        _pred("6.2", "pdpa_2010", score_band="0"),
    ]
    out = score_against_gold(pred, gold)
    assert out["false_zero_rate"] == 0.0


def test_false_zero_rate_partial() -> None:
    gold = [
        _gold("g4", "6.2", "service_tax_act_2018", score_band="0.5"),
        _gold("g5", "6.2", "income_tax_act_1967", score_band="0.5"),
    ]
    pred = [_pred("6.2", "service_tax_act_2018", score_band="0.5")]  # found 1 of 2
    out = score_against_gold(pred, gold)
    assert out["false_zero_rate"] == 0.5


# ─── citation fidelity (verbatim present) ────────────────────────────────────


def test_citation_fidelity_paraphrase_scores_zero() -> None:
    gold = [
        _gold(
            "g1",
            "6.2",
            "pdpa_2010",
            span_text="Personal data shall not be transferred outside Malaysia.",
            score_band="0",
        )
    ]
    pred = [
        _pred(
            "6.2",
            "pdpa_2010",
            span_text="The law generally restricts sending data abroad.",  # paraphrase
            score_band="0",
        )
    ]
    out = score_against_gold(pred, gold)
    assert out["citation_fidelity"] == 0.0
    # Still a valid provision match (precision/recall unaffected by paraphrase).
    assert out["recall"] == 1.0


def test_citation_fidelity_verbatim_substring_scores_one() -> None:
    gold = [
        _gold(
            "g1",
            "6.2",
            "pdpa_2010",
            span_text="shall not be transferred outside Malaysia",
            score_band="0",
        )
    ]
    pred = [
        _pred(
            "6.2",
            "pdpa_2010",
            span_text="Personal data shall not be transferred outside Malaysia under the Act.",
            score_band="0",
        )
    ]
    out = score_against_gold(pred, gold)
    assert out["citation_fidelity"] == 1.0


# ─── field accuracy (score_band + authority_tier) ────────────────────────────


def test_field_accuracy_requires_all_three_fields() -> None:
    gold = [_gold("g1", "6.2", "pdpa_2010", score_band="0", authority_tier="tier_1_statute")]
    # Correct indicator + verbatim, but WRONG score_band → field_accuracy 0.
    pred = [_pred("6.2", "pdpa_2010", score_band="1", authority_tier="tier_1_statute")]
    out = score_against_gold(pred, gold)
    assert out["indicator_accuracy"] == 1.0
    assert out["score_band_accuracy"] == 0.0
    assert out["field_accuracy"] == 0.0  # all-3 gate fails on band


def test_authority_tier_mismatch_lowers_field_accuracy() -> None:
    gold = [_gold("g1", "6.2", "pdpa_2010", score_band="0", authority_tier="tier_1_statute")]
    pred = [_pred("6.2", "pdpa_2010", score_band="0", authority_tier="tier_3_guideline")]
    out = score_against_gold(pred, gold)
    assert out["authority_tier_accuracy"] == 0.0
    assert out["field_accuracy"] == 0.0


# ─── empty / edge cases ──────────────────────────────────────────────────────


def test_empty_prediction_against_gold_recall_zero() -> None:
    gold = _malaysia_62_gold()
    out = score_against_gold([], gold)
    assert out["recall"] == 0.0
    assert out["counts"]["fn"] == 5
    # positive_gold = 2 (the two 0.5-band provisions); both missed → 2/2 = 1.0.
    assert out["false_zero_rate"] == 1.0


def test_empty_prediction_false_zero_rate_is_one_over_positives() -> None:
    gold = _malaysia_62_gold()  # 2 of 5 have score_band 0.5
    out = score_against_gold([], gold)
    assert out["false_zero_rate"] == 1.0


def test_empty_gold_yields_zero_metrics() -> None:
    out = score_against_gold([_pred("6.2", "x")], [])
    assert out["precision"] == 0.0
    assert out["recall"] == 0.0
    assert out["counts"]["gold_total"] == 0


def test_lenient_doc_id_normalisation_matches() -> None:
    """doc_id spelling drift (case / punctuation) still pairs via lenient pass."""
    gold = [_gold("g1", "6.2", "PDPA_2010", score_band="0")]
    pred = [_pred("6.2", "pdpa-2010", score_band="0")]
    out = score_against_gold(pred, gold)
    assert out["counts"]["tp"] == 1
    assert out["recall"] == 1.0
