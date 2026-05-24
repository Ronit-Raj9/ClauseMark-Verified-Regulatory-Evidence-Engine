"""Pure provenance helpers — sha256 + doc_id derivation.

Lives outside the service so it stays trivially testable with no I/O.
"""

from __future__ import annotations

import hashlib

from rie_contracts import SourceRegistryEntry


def compute_sha256(raw: bytes) -> str:
    """Return the hex sha256 digest of `raw`. 64 chars, lower-case."""
    return hashlib.sha256(raw).hexdigest()


def derive_doc_id(entry: SourceRegistryEntry) -> str:
    """`doc_id == source_id` — the registry is the system of record for IDs.

    Centralised so callers never compute their own id.
    """
    return entry.source_id
