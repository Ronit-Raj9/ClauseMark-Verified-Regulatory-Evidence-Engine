"""rie-ingest — local + remote document ingest adapter.

Implements `IngestPort` from `rie_contracts`. Bytes enter the trust boundary
here and are addressed by sha256 from this point on.
"""

from rie_ingest.cache import RawByteCache
from rie_ingest.discovery import (
    DiscoveryConfig,
    DiscoveryConfigError,
    DiscoveryCrawler,
    ProposedSource,
)
from rie_ingest.provenance import compute_sha256, derive_doc_id
from rie_ingest.service import IngestError, IngestService, TransientHTTPError

__all__ = [
    "DiscoveryConfig",
    "DiscoveryConfigError",
    "DiscoveryCrawler",
    "IngestError",
    "IngestService",
    "ProposedSource",
    "RawByteCache",
    "TransientHTTPError",
    "compute_sha256",
    "derive_doc_id",
]
