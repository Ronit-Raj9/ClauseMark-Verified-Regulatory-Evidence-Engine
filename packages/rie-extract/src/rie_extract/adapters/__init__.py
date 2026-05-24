"""Concrete extractor adapters — one per document family."""

from rie_extract.adapters.docling_extractor import DoclingExtractor, docling_available
from rie_extract.adapters.html_extractor import HtmlExtractor
from rie_extract.adapters.pymupdf_extractor import PyMuPdfExtractor
from rie_extract.adapters.text_extractor import TextExtractor
from rie_extract.adapters.vlm_ocr import (
    ScannedNotEnabledExtractor,
    VlmOcrExtractor,
)
from rie_extract.vlm_client import FakeVlmClient, OllamaVlmClient, VlmClient

__all__ = [
    "DoclingExtractor",
    "FakeVlmClient",
    "HtmlExtractor",
    "OllamaVlmClient",
    "PyMuPdfExtractor",
    "ScannedNotEnabledExtractor",
    "TextExtractor",
    "VlmClient",
    "VlmOcrExtractor",
    "docling_available",
]
