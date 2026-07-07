"""Per-language retrieval recall @ k for multilingual gold sets.

Groups gold items by language (typically ``DocumentMeta.language`` or a
gold-sidecar tag) and measures, for each language bucket, what fraction of
gold ``doc_id`` values appear in the top-*k* retrieved id lists aligned to
those gold items.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "LanguageRecall",
    "MultilingualRecallReport",
    "compute_multilingual_recall",
]


class LanguageRecall(BaseModel):
    """Recall @ k for one language bucket."""

    model_config = ConfigDict(frozen=True)

    language: str
    gold_count: int = Field(ge=0)
    hit_count: int = Field(ge=0)
    recall_at_k: float = Field(ge=0.0, le=1.0)


class MultilingualRecallReport(BaseModel):
    """Aggregated multilingual retrieval recall."""

    model_config = ConfigDict(frozen=True)

    per_language: list[LanguageRecall]
    macro_recall: float = Field(ge=0.0, le=1.0)


def _recalled(doc_id: str, retrieved_ids: Sequence[str], k: int) -> bool:
    top = retrieved_ids[:k] if k > 0 else retrieved_ids
    if doc_id in top:
        return True
    # Also accept substring hits — callers may supply element ids that embed doc_id.
    return any(doc_id in rid for rid in top)


def compute_multilingual_recall(
    gold_doc_ids: Sequence[str],
    gold_languages: Sequence[str],
    retrieved_per_query: Sequence[Sequence[str]],
    *,
    k: int = 8,
) -> MultilingualRecallReport:
    """Compute per-language recall @ ``k``.

    ``gold_doc_ids``, ``gold_languages``, and ``retrieved_per_query`` must be
    equal length and aligned row-wise (one retrieval list per gold item).
    """
    if not (len(gold_doc_ids) == len(gold_languages) == len(retrieved_per_query)):
        msg = (
            "length mismatch: gold_doc_ids="
            f"{len(gold_doc_ids)} gold_languages={len(gold_languages)} "
            f"retrieved={len(retrieved_per_query)}"
        )
        raise ValueError(msg)

    buckets: dict[str, list[tuple[str, Sequence[str]]]] = defaultdict(list)
    for doc_id, lang, retrieved in zip(
        gold_doc_ids, gold_languages, retrieved_per_query, strict=True
    ):
        buckets[lang.lower()].append((doc_id, retrieved))

    per_language: list[LanguageRecall] = []
    for lang in sorted(buckets):
        pairs = buckets[lang]
        hits = sum(1 for doc_id, retrieved in pairs if _recalled(doc_id, retrieved, k))
        total = len(pairs)
        recall = hits / total if total else 0.0
        per_language.append(
            LanguageRecall(
                language=lang,
                gold_count=total,
                hit_count=hits,
                recall_at_k=recall,
            )
        )

    macro = sum(lr.recall_at_k for lr in per_language) / len(per_language) if per_language else 0.0
    return MultilingualRecallReport(per_language=per_language, macro_recall=macro)
