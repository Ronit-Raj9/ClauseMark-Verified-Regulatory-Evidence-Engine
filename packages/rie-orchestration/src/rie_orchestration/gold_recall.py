"""Gold-set retrieval recall lookup for honest absence reasoning (§7).

Resolves measured recall per ``(pillar_id, indicator_id)`` from:

1. Optional ``CoverageReasonerPort.lookup_gold_recall`` on the wired adapter.
2. Best-effort hit overlap between gold items and in-run retrieval hits
   (same matching rules as ``rie_eval.Evaluator.measure_retrieval_recall``).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from rie_contracts import GoldItem, RetrievalHit

from rie_orchestration.state import RieState
from rie_orchestration.wiring import AdapterBundle

log = logging.getLogger(__name__)

__all__ = ["lookup_gold_recall"]


def lookup_gold_recall(
    bundle: AdapterBundle,
    pillar_id: str,
    indicator_id: str,
    *,
    state: RieState | None = None,
) -> float | None:
    """Return measured gold-set recall for an indicator, or ``None`` if unknown."""
    if state is not None and state.get("ingest_degraded"):
        return None

    port_lookup = getattr(bundle.coverage, "lookup_gold_recall", None)
    if callable(port_lookup):
        try:
            value = port_lookup(pillar_id=pillar_id, indicator_id=indicator_id)
            if value is not None:
                return float(value)
        except TypeError:
            try:
                value = port_lookup(pillar_id, indicator_id)
                if value is not None:
                    return float(value)
            except Exception:
                log.debug("coverage.lookup_gold_recall failed", exc_info=True)
        except Exception:
            log.debug("coverage.lookup_gold_recall failed", exc_info=True)

    if state is not None:
        return _recall_from_run_hits(bundle, pillar_id, indicator_id, state)

    return None


def _recall_from_run_hits(
    bundle: AdapterBundle,
    pillar_id: str,
    indicator_id: str,
    state: RieState,
) -> float | None:
    """Fraction of gold items whose doc/span appears in this run's retrieval hits."""
    try:
        gold = list(bundle.config.load_gold(pillar_id))
    except Exception:
        return None
    jurisdiction = state.get("jurisdiction", "")
    ind_gold = [
        g
        for g in gold
        if g.indicator_id == indicator_id and g.jurisdiction == jurisdiction
    ]
    if not ind_gold:
        return None
    hits = state.get("candidate_hits_by_pillar", {}).get(pillar_id, [])
    if not hits:
        return None
    retrieved = _flatten_retrieved(hits)
    hits_count = sum(1 for item in ind_gold if _gold_hit_in_retrieved(item, retrieved))
    return hits_count / len(ind_gold)


def _flatten_retrieved(hits: Sequence[RetrievalHit]) -> list[str]:
    retrieved: list[str] = []
    for hit in hits:
        retrieved.append(hit.parent_element_id)
        retrieved.extend(hit.neighbourhood_element_ids)
    return retrieved


def _gold_hit_in_retrieved(item: GoldItem, retrieved: Sequence[str]) -> bool:
    """Match rules aligned with ``rie_eval.service._gold_hit_in_retrieved``."""
    if not retrieved:
        return False
    needle_doc = item.doc_id.lower()
    needle_span = _normalise(item.span_text)
    for rid in retrieved:
        if rid is None:
            continue
        rid_norm = rid.lower()
        if needle_doc and needle_doc in rid_norm:
            return True
        if needle_span and needle_span in _normalise(rid):
            return True
    return False


def _normalise(text: str) -> str:
    return " ".join(text.split()).lower()
