"""Cross-corpus recall aggregation — Phase 2 multi-corpus absence evidence.

``systemArchitecture.md`` §1 flags "defensible absence scoring *beyond the
provided corpus*" as Phase 2. The §7 absence statement is only as strong as the
recall number attached to it; when an indicator has been searched across several
corpora / jurisdictions, the honest recall to attach is the *size-weighted*
combination of the per-corpus recalls — not a naive mean that would let a tiny,
well-covered corpus mask a large, poorly-covered one.

Pure data + arithmetic. No I/O, no model calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["CorpusRecall", "CrossCorpusRecall"]


@dataclass(frozen=True)
class CorpusRecall:
    """Measured gold-set recall for one corpus, with its size (weight)."""

    corpus_id: str
    recall: float
    size: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.recall <= 1.0:
            msg = f"recall must be in [0.0, 1.0], got {self.recall}"
            raise ValueError(msg)
        if self.size < 0:
            msg = f"size must be >= 0, got {self.size}"
            raise ValueError(msg)


@dataclass(frozen=True)
class CrossCorpusRecall:
    """Aggregate measured recall across multiple corpora for one indicator."""

    indicator_id: str
    per_corpus: list[CorpusRecall] = field(default_factory=list)

    @staticmethod
    def combine(recalls: list[CorpusRecall]) -> float:
        """Size-weighted mean of per-corpus recalls, clamped to ``[0.0, 1.0]``.

        Weighting rule: ``sum(r.recall * r.size) / sum(r.size)``. A corpus of
        size 0 contributes nothing. When every corpus has size 0 (or the list
        is empty) the result is ``0.0`` — the most conservative recall, which
        keeps the §7 absence statement honest rather than overstated.
        """
        total_size = sum(r.size for r in recalls)
        if total_size == 0:
            return 0.0
        weighted = sum(r.recall * r.size for r in recalls)
        combined = weighted / total_size
        # Defend against float drift past the bounds.
        return min(1.0, max(0.0, combined))

    def combined_recall(self) -> float:
        """Convenience: :meth:`combine` over this record's ``per_corpus``."""
        return self.combine(self.per_corpus)
