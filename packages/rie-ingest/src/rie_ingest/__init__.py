"""rie-ingest — local + remote document ingest adapter.

Implements `IngestPort` from `rie_contracts`. Bytes enter the trust boundary
here and are addressed by sha256 from this point on.
"""

from rie_ingest.cache import RawByteCache
from rie_ingest.crawler import SourceDiscoveryCrawler
from rie_ingest.discovery import (
    DiscoveryConfig,
    DiscoveryConfigError,
    DiscoveryCrawler,
    ProposedSource,
)
from rie_ingest.discovery_tag import (
    DiscoveryRecord,
    DiscoveryTagger,
    load_gold_for,
    normalize_indicator_id,
    tag_discovery,
    token_overlap,
)
from rie_ingest.economy_ingest import (
    EconomyDocument,
    EconomyIngestError,
    EconomyIngestor,
    ExtraMeta,
    available_economies,
    load_economy_documents,
)
from rie_ingest.provenance import compute_sha256, derive_doc_id
from rie_ingest.robots import RobotsPolicy
from rie_ingest.service import IngestError, IngestService, TransientHTTPError

__all__ = [
    "DiscoveryConfig",
    "DiscoveryConfigError",
    "DiscoveryCrawler",
    "DiscoveryRecord",
    "DiscoveryTagger",
    "EconomyDocument",
    "EconomyIngestError",
    "EconomyIngestor",
    "ExtraMeta",
    "IngestError",
    "IngestService",
    "ProposedSource",
    "RawByteCache",
    "RobotsPolicy",
    "SourceDiscoveryCrawler",
    "TransientHTTPError",
    "available_economies",
    "compute_sha256",
    "derive_doc_id",
    "load_economy_documents",
    "load_gold_for",
    "normalize_indicator_id",
    "tag_discovery",
    "token_overlap",
]
