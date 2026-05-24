"""Pure metric subpackage.

Re-exports the original ``rie_eval.metrics`` symbols (PRF/macro/confusion + the
``NO_CLAIM_LABEL`` sentinel) so existing call sites keep working, and adds:

- ``multilingual_recall``: per-language retrieval recall @ k.
- ``external_recall``:     fraction of gold items whose source doc is outside
                           the ingested corpus.
"""

from __future__ import annotations

from rie_eval.metrics.classification import NO_CLAIM_LABEL, confusion, macro_avg, prf
from rie_eval.metrics.external_recall import (
    ExternalRecallReport,
    compute_external_recall,
    load_out_of_corpus_sidecar,
)
from rie_eval.metrics.multilingual_recall import (
    LanguageRecall,
    MultilingualRecallReport,
    compute_multilingual_recall,
)

__all__ = [
    "NO_CLAIM_LABEL",
    "ExternalRecallReport",
    "LanguageRecall",
    "MultilingualRecallReport",
    "compute_external_recall",
    "compute_multilingual_recall",
    "confusion",
    "load_out_of_corpus_sidecar",
    "macro_avg",
    "prf",
]
