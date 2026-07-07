"""Indicator-id display aliasing.

The Round-1 RDTII gold database uses *decimal* indicator ids (e.g. 6 dot 4,
7 dot 2, 12 dot 10). The judge OUTPUT_TEMPLATE expects the *display* alias
form ``P{pillar}-I{indicator}`` (e.g. P6-I4, P7-I2, P12-I10).

Both directions are pure, deterministic, lossless string transforms. The
decimal id is always preserved in the JSON evidence package; the CSV carries
the display alias (judges validate the ``P{n}-I{m}`` shape programmatically).

Edge cases handled gracefully:

* Leading zeros — ``"06.04"`` -> ``"P6-I4"`` (numeric coercion strips them).
* 3-level decimals — ``"6.4.1"`` is treated as pillar ``6`` / indicator
  ``4.1`` is *rejected*; only two numeric levels are valid for the alias.
* Already-aliased input passed to :func:`to_display_indicator` is returned
  unchanged (idempotent), and likewise decimal input to :func:`to_decimal`.
"""

from __future__ import annotations

import re

_DISPLAY_RE = re.compile(r"^P(\d+)-I(\d+)$")
_DECIMAL_RE = re.compile(r"^(\d+)\.(\d+)$")


def to_display_indicator(decimal_id: str) -> str:
    """Convert a decimal indicator id to the ``P{pillar}-I{indicator}`` alias.

    Decimal pillar-dot-indicator maps to ``P{pillar}-I{indicator}`` (e.g.
    6 dot 4 -> P6-I4; 12 dot 10 -> P12-I10).

    Already-aliased ids are returned unchanged (idempotent). Leading zeros are
    stripped via integer coercion.
    """
    candidate = decimal_id.strip()
    if _DISPLAY_RE.match(candidate):
        # already a display alias — normalise (strip leading zeros) and return.
        m = _DISPLAY_RE.match(candidate)
        assert m is not None
        return f"P{int(m.group(1))}-I{int(m.group(2))}"
    m = _DECIMAL_RE.match(candidate)
    if m is None:
        msg = f"cannot parse indicator id {decimal_id!r}: expected '<pillar>.<indicator>'"
        raise ValueError(msg)
    pillar, indicator = int(m.group(1)), int(m.group(2))
    return f"P{pillar}-I{indicator}"


def to_decimal(display: str) -> str:
    """Inverse of :func:`to_display_indicator`.

    ``P{pillar}-I{indicator}`` maps back to pillar-dot-indicator (e.g.
    P6-I4 -> 6 dot 4; P12-I10 -> 12 dot 10).

    Already-decimal ids are returned normalised (idempotent).
    """
    candidate = display.strip()
    if _DECIMAL_RE.match(candidate):
        m = _DECIMAL_RE.match(candidate)
        assert m is not None
        return f"{int(m.group(1))}.{int(m.group(2))}"
    m = _DISPLAY_RE.match(candidate)
    if m is None:
        msg = f"cannot parse display indicator {display!r}: expected 'P<pillar>-I<indicator>'"
        raise ValueError(msg)
    pillar, indicator = int(m.group(1)), int(m.group(2))
    return f"{pillar}.{indicator}"


def is_display_indicator(value: str) -> bool:
    """True iff ``value`` matches the ``P{pillar}-I{indicator}`` display shape."""
    return _DISPLAY_RE.match(value.strip()) is not None
