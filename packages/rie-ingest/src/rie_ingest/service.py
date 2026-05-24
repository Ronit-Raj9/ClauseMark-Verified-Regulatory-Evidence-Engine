"""`IngestService` — local + remote document ingest, implements `IngestPort`.

This adapter is the system's only legitimate origin point for raw bytes. Every
downstream stage trusts the `DocumentMeta` produced here, so the invariants
matter:

- `doc_id == source_id` (the registry is the system of record).
- `sha256` is computed from the *bytes the rest of the pipeline will see*.
- `retrieved_at` is the moment the bytes entered our trust boundary.
- Bytes are cached by sha256 so a re-run never re-fetches.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import httpx
from rie_config import ConfigRepository
from rie_contracts import DocumentMeta, SourceRegistryEntry
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from rie_ingest.cache import RawByteCache
from rie_ingest.provenance import compute_sha256, derive_doc_id

# Extensions surfaced by `list_sample_laws`. Kept narrow on purpose — anything
# else needs a deliberate routing decision in `rie-extract`.
_SAMPLE_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".html", ".htm", ".txt"})

# Default HTTP timeout for remote sources. Conservative — government portals
# are slow; a too-eager timeout shows up as `insufficient_coverage`, which is
# the opposite of what we want for a known-good URL.
_DEFAULT_HTTP_TIMEOUT_SECONDS: float = 30.0


class IngestError(RuntimeError):
    """Raised when bytes cannot be obtained from any configured source."""


class TransientHTTPError(IngestError):
    """5xx-class error that tenacity should retry."""


def _is_transient_status(status: int) -> bool:
    # 408 (request timeout), 429 (too many requests), and 5xx are transient.
    # 4xx (except 408/429) is the caller's bug — never retry.
    return status == 408 or status == 429 or 500 <= status < 600


@dataclass
class IngestService:
    """Implements `IngestPort`. Hexagonal seam between the filesystem / network
    and the rest of the engine.
    """

    repo_root: Path
    config: ConfigRepository
    cache: RawByteCache = field(init=False)
    http_client: httpx.Client | None = None
    http_timeout: float = _DEFAULT_HTTP_TIMEOUT_SECONDS
    max_http_attempts: int = 3

    def __post_init__(self) -> None:
        self.cache = RawByteCache(root=self.repo_root / "data" / "cache" / "raw")

    # ─── Port methods ───────────────────────────────────────────────────────

    def load_source_registry(self, jurisdiction: str) -> Sequence[SourceRegistryEntry]:
        return self.config.load_source_registry(jurisdiction)

    def load_document_bytes(self, entry: SourceRegistryEntry) -> tuple[DocumentMeta, bytes]:
        raw = self._fetch_raw(entry)
        sha256 = compute_sha256(raw)
        # Persist into the cache so re-runs short-circuit. `store` is idempotent.
        self.cache.store(sha256, raw)
        meta = DocumentMeta(
            doc_id=derive_doc_id(entry),
            jurisdiction=entry.jurisdiction,
            title=entry.title,
            document_type=entry.document_type,
            effective_date=entry.effective_date,
            authority_tier=entry.authority_tier,
            source_url=entry.source_url,
            sha256=sha256,
            retrieved_at=datetime.now(UTC),
            language=entry.language,
        )
        return meta, raw

    def list_sample_laws(self, samples_dir: Path) -> Sequence[Path]:
        if not samples_dir.exists():
            return []
        out: list[Path] = [
            p
            for p in samples_dir.iterdir()
            if p.is_file() and p.suffix.lower() in _SAMPLE_EXTENSIONS
        ]
        out.sort()
        return out

    # ─── Internals ──────────────────────────────────────────────────────────

    def _fetch_raw(self, entry: SourceRegistryEntry) -> bytes:
        # 1. Local path wins — sample-law mode + offline reproducibility.
        if entry.local_path:
            local = (self.repo_root / entry.local_path).resolve()
            if not local.exists():
                raise IngestError(f"local_path declared but missing on disk: {entry.local_path}")
            raw = local.read_bytes()
            # Optional cache hop: if we already have these bytes by sha, skip
            # re-reading on later calls. Cheap and keeps behaviour consistent
            # with the HTTP path.
            sha = compute_sha256(raw)
            cached = self.cache.cached(sha)
            return cached if cached is not None else raw

        # 2. Otherwise HTTP. URL is mandatory if no local_path.
        if not entry.source_url:
            raise IngestError(f"entry {entry.source_id!r}: needs local_path or source_url")

        # Try a sha-keyed cache lookup ONLY if the entry advertises its own
        # sha256_hash — we cannot compute the key without first reading the
        # bytes, so we can only short-circuit the network call when the
        # registry has pinned the hash.
        if entry.sha256_hash:
            cached = self.cache.cached(entry.sha256_hash)
            if cached is not None:
                return cached

        return self._http_get_with_retry(entry.source_url)

    def _http_get_with_retry(self, url: str) -> bytes:
        # tenacity wraps `_http_get_once`. We keep the policy declarative so
        # the retry count + backoff are visible at the call site.
        @retry(
            reraise=True,
            stop=stop_after_attempt(self.max_http_attempts),
            wait=wait_exponential(multiplier=0.1, min=0.1, max=1.0),
            retry=retry_if_exception_type(
                (TransientHTTPError, httpx.TransportError, httpx.TimeoutException)
            ),
        )
        def _do() -> bytes:
            return self._http_get_once(url)

        try:
            return _do()
        except RetryError as exc:  # pragma: no cover — reraise=True bypasses this
            raise IngestError(f"GET {url} failed after retries") from exc

    def _http_get_once(self, url: str) -> bytes:
        client = self.http_client
        if client is None:
            # Per-call client — fine for the MVP. Callers wanting connection
            # pooling pass their own client (e.g. tests with respx).
            with httpx.Client(timeout=self.http_timeout) as c:
                resp = c.get(url)
        else:
            resp = client.get(url)

        if _is_transient_status(resp.status_code):
            raise TransientHTTPError(f"GET {url} -> {resp.status_code} (transient)")
        # Permanent failures: surface as plain IngestError so they do not retry.
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise IngestError(f"GET {url} -> {resp.status_code}") from exc
        return resp.content
