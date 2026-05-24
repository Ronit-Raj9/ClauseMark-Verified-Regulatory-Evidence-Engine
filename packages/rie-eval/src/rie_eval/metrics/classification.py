"""Pure metric functions for classification evaluation.

These are deterministic, vectorised over Python lists, and unit-tested with
hand-computed expected values. They never touch I/O.

Conventions:
- ``y_true`` and ``y_pred`` are aligned, equal-length sequences of label
  strings (typically ``indicator_id`` values such as ``"6.4"``).
- A reserved label ``NO_CLAIM_LABEL`` indicates "the system did not emit a
  claim for this gold item." Per-label PRF excludes ``NO_CLAIM_LABEL`` from
  the set of evaluated labels (a non-prediction is not its own class), but
  it does count as a false negative for the true label.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

# Sentinel used by the service when no model claim matches a gold item.
NO_CLAIM_LABEL: str = "__no_claim__"


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def prf(
    y_true: Sequence[str],
    y_pred: Sequence[str],
) -> dict[str, tuple[float, float, float]]:
    """Per-label precision/recall/F1.

    Returns a dict keyed by label -> (precision, recall, f1). The label set is
    the union of labels seen in either sequence, with ``NO_CLAIM_LABEL`` removed
    (it is a sentinel, not a real class).
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"length mismatch: y_true={len(y_true)} y_pred={len(y_pred)}")
    labels = (set(y_true) | set(y_pred)) - {NO_CLAIM_LABEL}
    out: dict[str, tuple[float, float, float]] = {}
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        out[label] = (precision, recall, f1)
    return out


def confusion(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[tuple[str, str], int]:
    """Return ``{(true_label, pred_label): count}`` over all observed pairs."""
    if len(y_true) != len(y_pred):
        raise ValueError(f"length mismatch: y_true={len(y_true)} y_pred={len(y_pred)}")
    counter: Counter[tuple[str, str]] = Counter()
    for t, p in zip(y_true, y_pred):
        counter[(t, p)] += 1
    return dict(counter)


def macro_avg(
    prf_per_label: dict[str, tuple[float, float, float]],
) -> tuple[float, float, float]:
    """Unweighted mean precision / recall / F1 across labels.

    Returns ``(0.0, 0.0, 0.0)`` for an empty input — caller decides what to do
    with that (typically "no gold items for this pillar").
    """
    if not prf_per_label:
        return (0.0, 0.0, 0.0)
    n = float(len(prf_per_label))
    p = sum(v[0] for v in prf_per_label.values()) / n
    r = sum(v[1] for v in prf_per_label.values()) / n
    f = sum(v[2] for v in prf_per_label.values()) / n
    return (p, r, f)


__all__ = ["NO_CLAIM_LABEL", "confusion", "macro_avg", "prf"]
