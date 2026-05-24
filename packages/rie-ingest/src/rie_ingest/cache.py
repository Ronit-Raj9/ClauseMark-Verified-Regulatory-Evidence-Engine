"""Content-addressable byte cache keyed by sha256.

A re-run that resolves the same `sha256` short-circuits the network / disk read.
This is the cheapest reproducibility lever in the pipeline and is exercised on
every ingest.

Layout: `<cache_root>/<sha256>.bin`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_HEX_LEN = 64


def _validate_sha256(sha256: str) -> None:
    if len(sha256) != _HEX_LEN:
        raise ValueError(f"sha256 must be {_HEX_LEN} hex chars, got {len(sha256)}")
    int(sha256, 16)  # raises ValueError on non-hex


@dataclass(frozen=True)
class RawByteCache:
    """A directory of `<sha256>.bin` files. Pure filesystem; no locking."""

    root: Path

    def cache_path(self, sha256: str) -> Path:
        _validate_sha256(sha256)
        return self.root / f"{sha256}.bin"

    def cached(self, sha256: str) -> bytes | None:
        path = self.cache_path(sha256)
        if not path.exists():
            return None
        return path.read_bytes()

    def store(self, sha256: str, raw: bytes) -> Path:
        path = self.cache_path(sha256)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic-ish write: write to .tmp then rename so a crash mid-write
        # never leaves a torn cache entry that masquerades as the real bytes.
        tmp = path.with_suffix(".bin.tmp")
        tmp.write_bytes(raw)
        tmp.replace(path)
        return path
