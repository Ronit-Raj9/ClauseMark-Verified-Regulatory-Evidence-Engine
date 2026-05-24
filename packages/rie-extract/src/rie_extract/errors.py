"""Typed errors for the extraction package."""

from __future__ import annotations


class ExtractionError(Exception):
    """Raised when an extractor cannot produce a usable element stream."""
