"""External-corpus recall — gold items whose sources sit outside what we ingested.

The frozen ``GoldItem`` contract forbids extra fields, so out-of-corpus
annotations live in a YAML sidecar validated against
``gold/_schema/out_of_corpus.schema.json``. This module loads that sidecar
and computes the fraction of gold items whose ``doc_id`` is absent from the
ingested corpus (or explicitly flagged ``out_of_corpus: true``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field
from rie_contracts import GoldItem

__all__ = [
    "ExternalRecallReport",
    "compute_external_recall",
    "load_out_of_corpus_sidecar",
]


class ExternalRecallReport(BaseModel):
    """Summary of how much gold depends on sources beyond the ingested corpus."""

    model_config = ConfigDict(frozen=True)

    total_gold: int = Field(ge=0)
    out_of_corpus_count: int = Field(ge=0)
    external_recall: float = Field(ge=0.0, le=1.0)
    """Fraction of gold items whose authoritative source is outside the ingested corpus."""


def load_out_of_corpus_sidecar(path: Path) -> dict[str, bool]:
    """Load ``gold_id -> out_of_corpus`` from a pillar sidecar YAML file.

    Returns an empty mapping when the file does not exist — sidecars are
    optional per pillar.
    """
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        msg = f"{path}: top-level YAML must be a mapping"
        raise ValueError(msg)
    data = cast(dict[str, Any], raw)
    items = data.get("items")
    if items is None:
        return {}
    if not isinstance(items, list):
        msg = f"{path}: `items` must be a list"
        raise ValueError(msg)
    out: dict[str, bool] = {}
    for entry in items:
        if not isinstance(entry, dict):
            continue
        entry_d = cast(dict[str, Any], entry)
        gold_id = entry_d.get("gold_id")
        flag = entry_d.get("out_of_corpus")
        if isinstance(gold_id, str) and isinstance(flag, bool):
            out[gold_id] = flag
    return out


def _is_out_of_corpus(
    item: GoldItem,
    ingested_doc_ids: set[str],
    sidecar: Mapping[str, bool],
) -> bool:
    if sidecar.get(item.gold_id) is True:
        return True
    if sidecar.get(item.gold_id) is False:
        return False
    return item.doc_id not in ingested_doc_ids


def compute_external_recall(
    gold_items: Sequence[GoldItem],
    *,
    ingested_doc_ids: set[str],
    sidecar: Mapping[str, bool] | None = None,
) -> ExternalRecallReport:
    """Compute the external-corpus fraction for a gold set.

    An item counts as out-of-corpus when:

    * its sidecar entry has ``out_of_corpus: true``, or
    * its ``doc_id`` is not present in ``ingested_doc_ids`` (and the sidecar
      does not explicitly mark it in-corpus).
    """
    side = sidecar or {}
    total = len(gold_items)
    if total == 0:
        return ExternalRecallReport(
            total_gold=0,
            out_of_corpus_count=0,
            external_recall=0.0,
        )
    out_count = sum(1 for item in gold_items if _is_out_of_corpus(item, ingested_doc_ids, side))
    return ExternalRecallReport(
        total_gold=total,
        out_of_corpus_count=out_count,
        external_recall=out_count / total,
    )
