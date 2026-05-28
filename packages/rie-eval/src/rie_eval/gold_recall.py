"""Gold-set retrieval recall for §7 honest absence reasoning.

Computes the fraction of gold items for a ``(jurisdiction, indicator)``
pair whose ``doc_id`` or ``span_text`` appears in the retrieved id/snippet
lists produced by the retrieval stage.

This module is the canonical implementation consumed by orchestration
(``rie_orchestration.gold_recall.lookup_gold_recall``) and by
:class:`~rie_eval.service.Evaluator`.
"""

from __future__ import annotations

from collections.abc import Sequence

from rie_contracts import GoldItem

__all__ = [
    "compute_gold_set_recall",
    "compute_indicator_gold_recall",
    "filter_gold_for_indicator",
    "gold_hit_in_retrieved",
    "normalise_span_text",
]


def normalise_span_text(text: str) -> str:
    """Collapse whitespace and lower-case for substring matching."""
    return " ".join(text.split()).lower()


def gold_hit_in_retrieved(item: GoldItem, retrieved: Sequence[str]) -> bool:
    """Return True when a gold item is recalled in ``retrieved``.

    A hit counts when either:

    * any retrieved id contains the gold ``doc_id`` (element ids often embed
      the doc id), or
    * the normalised gold ``span_text`` appears as a substring of any retrieved
      id/snippet (callers may pass plain-text snippets).
    """
    if not retrieved:
        return False
    needle_doc = item.doc_id.lower()
    needle_span = normalise_span_text(item.span_text)
    for rid in retrieved:
        if rid is None:
            continue
        rid_norm = rid.lower()
        if needle_doc and needle_doc in rid_norm:
            return True
        if needle_span and needle_span in normalise_span_text(rid):
            return True
    return False


def filter_gold_for_indicator(
    gold: Sequence[GoldItem],
    *,
    indicator_id: str,
    jurisdiction: str,
) -> list[GoldItem]:
    """Gold items scoped to one ``(jurisdiction, indicator_id)`` pair."""
    return [
        g
        for g in gold
        if g.indicator_id == indicator_id and g.jurisdiction == jurisdiction
    ]


def compute_indicator_gold_recall(
    gold: Sequence[GoldItem],
    *,
    indicator_id: str,
    jurisdiction: str,
    retrieved: Sequence[str],
) -> float | None:
    """Measured recall for one indicator within a jurisdiction.

    Returns ``None`` when:

    * no gold items exist for the pair, or
    * ``retrieved`` is empty (recall cannot be measured without retrieval output).

    Otherwise returns ``hits / len(ind_gold)`` in ``[0.0, 1.0]``.
    """
    ind_gold = filter_gold_for_indicator(
        gold, indicator_id=indicator_id, jurisdiction=jurisdiction
    )
    if not ind_gold or not retrieved:
        return None
    hits = sum(1 for item in ind_gold if gold_hit_in_retrieved(item, retrieved))
    return hits / len(ind_gold)


def compute_gold_set_recall(
    gold: Sequence[GoldItem],
    retrieved_per_query: Sequence[Sequence[str]],
) -> float:
    """Pillar-level recall with one retrieved-id list per gold item.

    ``retrieved_per_query`` must be the same length as ``gold`` and aligned
    row-wise (one retrieval list per gold annotation). Returns ``0.0`` when
    ``gold`` is empty.
    """
    if not gold:
        return 0.0
    if len(retrieved_per_query) != len(gold):
        msg = (
            f"retrieved_per_query length {len(retrieved_per_query)} "
            f"!= gold length {len(gold)}"
        )
        raise ValueError(msg)
    hits = sum(
        1
        for item, retrieved in zip(gold, retrieved_per_query, strict=True)
        if gold_hit_in_retrieved(item, retrieved)
    )
    return hits / len(gold)
