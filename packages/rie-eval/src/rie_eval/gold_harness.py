"""Gold harness — score predicted provisions against the Round-1 gold DB.

This is the **Substantive-Accuracy (40%)** measurement surface. Unlike
``service.Evaluator`` (which scores ``Claim`` objects out of the live graph),
the harness scores a *flat, serialisable* list of predicted-provision records
against the ESCAP Round-1 gold set so the tool can be benchmarked offline and
the result reproduced by a judge from a single ``--predicted preds.json`` file.

Why match per ``(jurisdiction, indicator_id, doc_id)``
------------------------------------------------------
One RDTII indicator maps to **many** distinct law provisions. E.g. Malaysia
indicator ``6.2`` (local-storage requirements) has FIVE gold rows — PDPA 2010,
the banking-sector code, the comms-sector code, the Service Tax Act 2018, and
the Income Tax Act 1967 — each a separate ``doc_id`` carrying its own
``score_band``. Scoring per *indicator* would collapse those five provisions
into one and reward a tool that found a single law. We therefore key the
matching on the triple ``(jurisdiction, indicator_id, doc_id)`` so precision /
recall / F1 are measured **per (indicator, law) provision pair**.

Predicted-input record shape
----------------------------
``score_against_gold`` / ``match_provisions`` accept ``predicted`` as a list
of plain ``dict`` records (e.g. parsed from the tool's output JSON). Each
record carries:

================  =========================================================
key               meaning
================  =========================================================
``jurisdiction``  UN economy name, e.g. ``"Malaysia"`` (matched casefolded).
``indicator_id``  RDTII decimal id as a STRING, e.g. ``"6.2"`` / ``"7.5"``.
``doc_id``        Stable law identifier — the unit of a provision. Compared
                  to ``GoldItem.doc_id`` (exact, then normalised fallback).
``span_text``     Verbatim quoted clause text (drives citation fidelity).
``score_band``    One of ``{"0", "0.5", "1", "null"}`` (``ScoreBand`` values).
``authority_tier`` one of the ``AuthorityTier`` values, e.g.
                  ``"tier_1_statute"``.
``discovery_tag`` ``"KNOWN"`` (claims a gold provision) or ``"NEW"`` (an
                  independent find beyond the sample kit). Drives
                  ``discovery_new_count``.
================  =========================================================

All keys are read defensively via ``.get`` so a partial record never raises.
Missing values fall back to empty string / ``None``.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from rie_contracts import GoldItem

from rie_eval.metrics import NO_CLAIM_LABEL, macro_avg, prf

# ─── Predicted-record key constants (documented above) ──────────────────────

K_JURISDICTION = "jurisdiction"
K_INDICATOR = "indicator_id"
K_DOC = "doc_id"
K_SPAN = "span_text"
K_SCORE_BAND = "score_band"
K_AUTHORITY = "authority_tier"
K_DISCOVERY = "discovery_tag"

DISCOVERY_KNOWN = "KNOWN"
DISCOVERY_NEW = "NEW"

# A provision is identified by this triple. One indicator → many provisions.
ProvisionKey = tuple[str, str, str]  # (jurisdiction, indicator_id, doc_id)


# ─── Normalisation helpers ──────────────────────────────────────────────────


def _norm_juris(value: object) -> str:
    """Case-insensitive, whitespace-collapsed jurisdiction key."""
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


def _norm_indicator(value: object) -> str:
    """Normalise a decimal indicator id.

    openpyxl-derived gold may carry floats (``6.1``) while predicted JSON
    carries strings (``"6.1"``). Coerce to a canonical string. We keep the
    textual form (``"6.10"`` ≠ ``"6.1"``) because trailing zeros are
    semantically distinct in RDTII (4.1 vs 4.10 collide as floats — see gold
    notes). A float input is rendered without a spurious ``.0``.
    """
    if isinstance(value, float):
        # 6.0 → "6"; 6.5 → "6.5". Avoid float repr noise.
        if value.is_integer():
            return str(int(value))
        return repr(value)
    return str(value or "").strip()


def _norm_doc(value: object) -> str:
    """Normalise a doc id for the *fallback* (lenient) comparison.

    Lowercase, strip non-alphanumerics. Used only when an exact ``doc_id``
    match fails, so two spellings of the same law id still pair.
    """
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def _provision_key(juris: object, indicator: object, doc: object) -> ProvisionKey:
    return (_norm_juris(juris), _norm_indicator(indicator), _norm_doc(doc))


def _verbatim_present(pred_span: str, gold_span: str) -> bool:
    """Citation fidelity: is the gold clause verbatim-present in the prediction?

    True when the (whitespace-normalised) gold span is a substring of the
    (whitespace-normalised) predicted span, OR vice-versa (the tool may quote
    a tighter sub-clause). Paraphrase → False (no point given). Empty gold or
    empty prediction → False.
    """
    p = re.sub(r"\s+", " ", (pred_span or "").strip()).casefold()
    g = re.sub(r"\s+", " ", (gold_span or "").strip()).casefold()
    if not p or not g:
        return False
    return g in p or p in g


# ─── Match record ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GoldMatch:
    """One resolved provision-pair outcome.

    Exactly one of the three states holds:

    - ``matched``           — both ``gold`` and ``predicted`` set (true positive).
    - ``missed``            — ``gold`` set, ``predicted`` None (false negative).
    - ``spurious``          — ``predicted`` set, ``gold`` None (false positive).

    Field-level booleans (``indicator_match`` etc.) are only meaningful when
    ``matched`` is True; they are False otherwise.
    """

    key: ProvisionKey
    gold: GoldItem | None
    predicted: Mapping[str, object] | None
    indicator_match: bool = False
    score_band_match: bool = False
    authority_tier_match: bool = False
    verbatim_present: bool = False

    @property
    def matched(self) -> bool:
        return self.gold is not None and self.predicted is not None

    @property
    def missed(self) -> bool:
        return self.gold is not None and self.predicted is None

    @property
    def spurious(self) -> bool:
        return self.gold is None and self.predicted is not None


@dataclass
class MatchSet:
    """The full matching outcome — TP / FN / FP partitions plus discovery."""

    matched: list[GoldMatch] = field(default_factory=list)
    missed: list[GoldMatch] = field(default_factory=list)
    spurious: list[GoldMatch] = field(default_factory=list)
    discovery_new: list[Mapping[str, object]] = field(default_factory=list)

    def all_(self) -> list[GoldMatch]:
        return [*self.matched, *self.missed, *self.spurious]


# ─── Matching ───────────────────────────────────────────────────────────────


def match_provisions(
    predicted: Sequence[Mapping[str, object]],
    gold: Sequence[GoldItem],
) -> MatchSet:
    """Match predicted provisions to gold, keyed by (jurisdiction, indicator, doc).

    The matching is **per-provision** (one indicator → many laws). For each
    gold item we look for a predicted record sharing the exact provision key;
    if none, a *lenient* pass pairs any still-unused predicted record whose
    ``doc_id`` normalises equal AND whose ``(jurisdiction, indicator)`` match
    (handles doc-id spelling drift). Each predicted record is consumed at most
    once.

    Predicted records tagged ``discovery_tag == "NEW"`` that do NOT pair with
    any gold item are collected in ``MatchSet.discovery_new`` (the
    Discovery-of-New-Evidence differentiator) and are NOT counted as spurious
    false positives — a NEW find beyond the sample kit is a feature, not an
    error. A ``NEW``-tagged record that DOES match a gold provision is treated
    as a normal match (the tool mislabelled a known provision as new).
    """
    # Index predicted by exact key and by lenient key; preserve order/dupes.
    by_exact: dict[ProvisionKey, list[int]] = defaultdict(list)
    by_lenient: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for i, rec in enumerate(predicted):
        ek = _provision_key(rec.get(K_JURISDICTION), rec.get(K_INDICATOR), rec.get(K_DOC))
        by_exact[ek].append(i)
        lk = (
            _norm_juris(rec.get(K_JURISDICTION)),
            _norm_indicator(rec.get(K_INDICATOR)),
            _norm_doc(rec.get(K_DOC)),
        )
        by_lenient[lk].append(i)

    used: set[int] = set()
    result = MatchSet()

    for g in gold:
        gk = _provision_key(g.jurisdiction, g.indicator_id, g.doc_id)
        chosen: int | None = None
        # Exact provision-key pass.
        for idx in by_exact.get(gk, []):
            if idx not in used:
                chosen = idx
                break
        # Lenient fallback — same (juris, indicator, normalised-doc).
        if chosen is None:
            lk = (_norm_juris(g.jurisdiction), _norm_indicator(g.indicator_id), _norm_doc(g.doc_id))
            for idx in by_lenient.get(lk, []):
                if idx not in used:
                    chosen = idx
                    break
        if chosen is None:
            result.missed.append(GoldMatch(key=gk, gold=g, predicted=None))
            continue
        used.add(chosen)
        rec = predicted[chosen]
        result.matched.append(
            GoldMatch(
                key=gk,
                gold=g,
                predicted=rec,
                indicator_match=(
                    _norm_indicator(rec.get(K_INDICATOR)) == _norm_indicator(g.indicator_id)
                ),
                score_band_match=(
                    str(rec.get(K_SCORE_BAND) or "").strip() == str(g.expected_score_band)
                ),
                authority_tier_match=(
                    str(rec.get(K_AUTHORITY) or "").strip() == str(g.expected_authority_tier)
                ),
                verbatim_present=_verbatim_present(str(rec.get(K_SPAN) or ""), g.span_text),
            )
        )

    # Unmatched predicted records → spurious OR discovery_new.
    for i, rec in enumerate(predicted):
        if i in used:
            continue
        tag = str(rec.get(K_DISCOVERY) or "").strip().upper()
        if tag == DISCOVERY_NEW:
            result.discovery_new.append(rec)
        else:
            ek = _provision_key(rec.get(K_JURISDICTION), rec.get(K_INDICATOR), rec.get(K_DOC))
            result.spurious.append(GoldMatch(key=ek, gold=None, predicted=rec))

    return result


# ─── Scoring ────────────────────────────────────────────────────────────────


def score_against_gold(
    predicted: Sequence[Mapping[str, object]],
    gold: Sequence[GoldItem],
) -> dict[str, object]:
    """Score predicted provisions against gold; macro + per-indicator breakdown.

    Returns a structured dict::

        {
          "precision": float, "recall": float, "f1": float,   # macro (per-provision)
          "field_accuracy": float,        # mean over matched of (indicator
                                          #   & score_band & authority_tier) all-3
          "indicator_accuracy": float,    # fraction of matched w/ indicator_id match
          "score_band_accuracy": float,
          "authority_tier_accuracy": float,
          "citation_fidelity": float,     # fraction of matched w/ gold span verbatim-present
          "false_zero_rate": float,       # fraction of gold provisions w/ a score>0
                                          #   that the tool missed entirely (false 0/absence)
          "discovery_new_count": int,     # predicted NEW not in gold
          "counts": {tp, fn, fp, gold_total, predicted_total},
          "per_indicator": {ind: {precision, recall, f1, support,
                                  field_accuracy, citation_fidelity}},
        }

    Precision / recall / F1 are computed over the **provision-pair** universe:
    each gold item and each (non-discovery) predicted record is one unit, keyed
    by ``(jurisdiction, indicator_id, doc_id)``. A matched pair where the
    predicted ``indicator_id`` disagrees with gold is a *wrong-indicator* hit —
    it lowers both that gold indicator's recall (missed under the true label)
    and the predicted indicator's precision (a false positive under the wrong
    label), exactly like ``service.Evaluator``.
    """
    ms = match_provisions(predicted, gold)

    # ── Per-provision PRF via the shared classification metric ──────────────
    # Build aligned y_true / y_pred sequences over the provision universe.
    # Each gold provision contributes its true indicator; the matched
    # prediction contributes its (possibly wrong) predicted indicator.
    y_true: list[str] = []
    y_pred: list[str] = []
    for m in ms.matched:
        assert m.gold is not None and m.predicted is not None
        y_true.append(_norm_indicator(m.gold.indicator_id))
        y_pred.append(_norm_indicator(m.predicted.get(K_INDICATOR)))
    for m in ms.missed:
        assert m.gold is not None
        y_true.append(_norm_indicator(m.gold.indicator_id))
        y_pred.append(NO_CLAIM_LABEL)
    for m in ms.spurious:
        assert m.predicted is not None
        y_true.append(NO_CLAIM_LABEL)
        y_pred.append(_norm_indicator(m.predicted.get(K_INDICATOR)))

    per_label = prf(y_true, y_pred)
    p_macro, r_macro, f_macro = macro_avg(per_label)

    # ── Field accuracy + citation fidelity over matched pairs ───────────────
    n_matched = len(ms.matched)
    if n_matched:
        field_acc = (
            sum(
                1
                for m in ms.matched
                if m.indicator_match and m.score_band_match and m.authority_tier_match
            )
            / n_matched
        )
        indicator_acc = sum(1 for m in ms.matched if m.indicator_match) / n_matched
        band_acc = sum(1 for m in ms.matched if m.score_band_match) / n_matched
        tier_acc = sum(1 for m in ms.matched if m.authority_tier_match) / n_matched
        citation_fidelity = sum(1 for m in ms.matched if m.verbatim_present) / n_matched
    else:
        field_acc = indicator_acc = band_acc = tier_acc = citation_fidelity = 0.0

    # ── False-zero rate ─────────────────────────────────────────────────────
    # A "false zero" = the tool emitted nothing for a gold provision that the
    # gold set records as restrictive (score_band != "0"/"null"). I.e. the tool
    # silently treated a real, scored provision as absent. Denominator = gold
    # provisions with a positive (>0) expected band.
    positive_gold = [g for g in gold if str(g.expected_score_band) not in ("0", "null", str(None))]
    if positive_gold:
        missed_keys = {m.key for m in ms.missed}
        false_zeros = sum(
            1
            for g in positive_gold
            if _provision_key(g.jurisdiction, g.indicator_id, g.doc_id) in missed_keys
        )
        false_zero_rate = false_zeros / len(positive_gold)
    else:
        false_zero_rate = 0.0

    # ── Per-indicator breakdown ─────────────────────────────────────────────
    per_indicator = _per_indicator_breakdown(ms)

    return {
        "precision": p_macro,
        "recall": r_macro,
        "f1": f_macro,
        "field_accuracy": field_acc,
        "indicator_accuracy": indicator_acc,
        "score_band_accuracy": band_acc,
        "authority_tier_accuracy": tier_acc,
        "citation_fidelity": citation_fidelity,
        "false_zero_rate": false_zero_rate,
        "discovery_new_count": len(ms.discovery_new),
        "counts": {
            "tp": n_matched,
            "fn": len(ms.missed),
            "fp": len(ms.spurious),
            "gold_total": len(gold),
            "predicted_total": len(predicted),
        },
        "per_indicator": per_indicator,
    }


def _per_indicator_breakdown(ms: MatchSet) -> dict[str, dict[str, float]]:
    """PRF + field/citation accuracy grouped by the TRUE indicator id.

    Precision/recall/F1 here are *provision-counting* per indicator: for an
    indicator ``i`` over its provision universe,

    - TP = matched pairs whose gold indicator is ``i`` AND predicted indicator
      is also ``i`` (correct provision under correct label),
    - FN = (gold provisions of ``i`` that were missed) + (matched pairs whose
      gold indicator is ``i`` but predicted indicator ≠ ``i``),
    - FP = (spurious predictions labelled ``i``) + (matched pairs whose
      predicted indicator is ``i`` but gold indicator ≠ ``i``).
    """
    indicators: set[str] = set()
    for m in ms.matched:
        assert m.gold is not None and m.predicted is not None
        indicators.add(_norm_indicator(m.gold.indicator_id))
        indicators.add(_norm_indicator(m.predicted.get(K_INDICATOR)))
    for m in ms.missed:
        assert m.gold is not None
        indicators.add(_norm_indicator(m.gold.indicator_id))
    for m in ms.spurious:
        assert m.predicted is not None
        indicators.add(_norm_indicator(m.predicted.get(K_INDICATOR)))
    indicators.discard(NO_CLAIM_LABEL)
    indicators.discard("")

    return {ind: _indicator_row(ms, ind) for ind in sorted(indicators)}


def _indicator_row(ms: MatchSet, ind: str) -> dict[str, float]:
    """Provision-counting PRF + field/citation accuracy for one indicator id."""
    tp = sum(1 for m in ms.matched if _gi(m) == ind and _pi(m) == ind)
    fn_matched = sum(1 for m in ms.matched if _gi(m) == ind and _pi(m) != ind)
    fp_matched = sum(1 for m in ms.matched if _pi(m) == ind and _gi(m) != ind)
    fn_missed = sum(1 for m in ms.missed if _gi(m) == ind)
    fp_spurious = sum(1 for m in ms.spurious if _pi(m) == ind)
    fn = fn_matched + fn_missed
    fp = fp_matched + fp_spurious
    support = tp + fn_matched + fn_missed

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    correct_pairs = [m for m in ms.matched if _gi(m) == ind and _pi(m) == ind]
    if correct_pairs:
        field_acc = sum(
            1 for m in correct_pairs if m.score_band_match and m.authority_tier_match
        ) / len(correct_pairs)
        citation = sum(1 for m in correct_pairs if m.verbatim_present) / len(correct_pairs)
    else:
        field_acc = citation = 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": float(support),
        "field_accuracy": field_acc,
        "citation_fidelity": citation,
    }


def _gi(m: GoldMatch) -> str:
    """Normalised gold indicator id of a match (``""`` when no gold)."""
    return _norm_indicator(m.gold.indicator_id) if m.gold is not None else ""


def _pi(m: GoldMatch) -> str:
    """Normalised predicted indicator id of a match (``""`` when no prediction)."""
    return _norm_indicator(m.predicted.get(K_INDICATOR)) if m.predicted is not None else ""


__all__ = [
    "DISCOVERY_KNOWN",
    "DISCOVERY_NEW",
    "GoldMatch",
    "MatchSet",
    "ProvisionKey",
    "match_provisions",
    "score_against_gold",
]
