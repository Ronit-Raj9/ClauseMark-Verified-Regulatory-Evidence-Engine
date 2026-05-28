"""rie-extract — document extraction + structure-graph builder.

Public entrypoint is `ExtractionService` which implements
`rie_contracts.DocumentExtractorPort`. Extraction is routed per document
type (PDF → PyMuPDF then Docling per §5.2, HTML → BeautifulSoup,
plain text → text splitter). Docling enriches layout/structure when
available and degrades gracefully to PyMuPDF-only. Scanned / image-only PDFs route
to the VLM-OCR adapter when enabled, otherwise to a §11
graceful-degradation stub.
"""

from rie_extract.errors import ExtractionError
from rie_extract.router import AdapterRouter, pick_adapter
from rie_extract.service import ExtractionService
from rie_extract.structure.graph_builder import StructureGraphBuilder
from rie_extract.structure.legal_numbering import (
    find_cross_references,
    parse_numbering,
)
from rie_extract.vlm_client import (
    FakeVlmClient,
    OllamaVlmClient,
    VlmClient,
)

__all__ = [
    "AdapterRouter",
    "ExtractionError",
    "ExtractionService",
    "FakeVlmClient",
    "OllamaVlmClient",
    "StructureGraphBuilder",
    "VlmClient",
    "find_cross_references",
    "parse_numbering",
    "pick_adapter",
]
