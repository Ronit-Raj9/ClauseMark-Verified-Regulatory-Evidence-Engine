"""Indicator-id display aliasing round-trips: decimal <-> P{n}-I{m}."""

from __future__ import annotations

import pytest
from rie_output.indicator_id import (
    is_display_indicator,
    to_decimal,
    to_display_indicator,
)

CASES = [
    ("6.4", "P6-I4"),
    ("7.2", "P7-I2"),
    ("12.10", "P12-I10"),
    ("6.1", "P6-I1"),
    ("7.5", "P7-I5"),
]


@pytest.mark.parametrize(("decimal", "display"), CASES)
def test_to_display(decimal: str, display: str) -> None:
    assert to_display_indicator(decimal) == display


@pytest.mark.parametrize(("decimal", "display"), CASES)
def test_to_decimal(decimal: str, display: str) -> None:
    assert to_decimal(display) == decimal


@pytest.mark.parametrize(("decimal", "display"), CASES)
def test_round_trip_decimal(decimal: str, display: str) -> None:
    assert to_decimal(to_display_indicator(decimal)) == decimal


@pytest.mark.parametrize(("decimal", "display"), CASES)
def test_round_trip_display(decimal: str, display: str) -> None:
    assert to_display_indicator(to_decimal(display)) == display


def test_leading_zeros_normalised() -> None:
    assert to_display_indicator("06.04") == "P6-I4"
    assert to_decimal("P06-I04") == "6.4"


def test_idempotent_display_input() -> None:
    assert to_display_indicator("P6-I4") == "P6-I4"


def test_idempotent_decimal_input() -> None:
    assert to_decimal("6.4") == "6.4"


@pytest.mark.parametrize("bad", ["6", "abc", "6.4.1", "P6", "I4", ""])
def test_to_display_rejects_garbage(bad: str) -> None:
    with pytest.raises(ValueError):
        to_display_indicator(bad)


@pytest.mark.parametrize("bad", ["P6", "abc", "6", "P6-4", ""])
def test_to_decimal_rejects_garbage(bad: str) -> None:
    with pytest.raises(ValueError):
        to_decimal(bad)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("P6-I4", True),
        ("P12-I10", True),
        ("6.4", False),
        ("garbage", False),
    ],
)
def test_is_display_indicator(value: str, expected: bool) -> None:
    assert is_display_indicator(value) is expected
