"""Unit tests for pure metric helpers."""

from __future__ import annotations

import pytest
from rie_eval.metrics import NO_CLAIM_LABEL, confusion, macro_avg, prf


def test_prf_perfect() -> None:
    y_true = ["6.4", "7.2", "7.4", "6.4"]
    y_pred = ["6.4", "7.2", "7.4", "6.4"]
    out = prf(y_true, y_pred)
    for label, (p, r, f) in out.items():
        assert p == 1.0, label
        assert r == 1.0, label
        assert f == 1.0, label


def test_prf_one_swap() -> None:
    # Swap one prediction: gold 7.2 → predicted 6.4
    y_true = ["6.4", "7.2", "7.4"]
    y_pred = ["6.4", "6.4", "7.4"]
    out = prf(y_true, y_pred)
    # 6.4: tp=1, fp=1, fn=0 → P=0.5, R=1.0, F1=2/3
    p, r, f = out["6.4"]
    assert p == pytest.approx(0.5)
    assert r == pytest.approx(1.0)
    assert f == pytest.approx(2 / 3)
    # 7.2: tp=0, fp=0, fn=1 → all zero
    assert out["7.2"] == (0.0, 0.0, 0.0)
    # 7.4: tp=1, fp=0, fn=0 → all one
    assert out["7.4"] == (1.0, 1.0, 1.0)


def test_prf_excludes_sentinel_label() -> None:
    y_true = ["6.4", NO_CLAIM_LABEL]
    y_pred = ["6.4", "7.2"]
    out = prf(y_true, y_pred)
    # The sentinel must NOT appear as its own label.
    assert NO_CLAIM_LABEL not in out
    # But it still affects 7.2's stats: pred=7.2 but true=sentinel → FP for 7.2.
    p, r, _ = out["7.2"]
    assert p == 0.0  # 0/1
    assert r == 0.0


def test_prf_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        prf(["a"], ["a", "b"])


def test_confusion_counts() -> None:
    y_true = ["a", "a", "b", "b"]
    y_pred = ["a", "b", "b", "b"]
    out = confusion(y_true, y_pred)
    assert out[("a", "a")] == 1
    assert out[("a", "b")] == 1
    assert out[("b", "b")] == 2


def test_confusion_symmetry_on_perfect_predictions() -> None:
    # Perfect prediction → confusion matrix lives entirely on the diagonal.
    labels = ["x", "y", "x", "z"]
    out = confusion(labels, labels)
    for (t, p), _ in out.items():
        assert t == p, f"off-diagonal entry on perfect predictions: ({t},{p})"


def test_confusion_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        confusion(["a"], ["a", "b"])


def test_macro_avg_basic() -> None:
    per_label = {"a": (1.0, 1.0, 1.0), "b": (0.0, 0.0, 0.0)}
    p, r, f = macro_avg(per_label)
    assert p == 0.5
    assert r == 0.5
    assert f == 0.5


def test_macro_avg_empty() -> None:
    assert macro_avg({}) == (0.0, 0.0, 0.0)
