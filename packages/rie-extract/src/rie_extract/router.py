"""Pick the right extractor for a `DocumentMeta` / file."""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from rie_contracts import DocumentMeta, Element, StructureEdge

from rie_extract.adapters.born_digital_pdf_extractor import BornDigitalPdfExtractor
from rie_extract.adapters.html_extractor import HtmlExtractor
from rie_extract.adapters.text_extractor import TextExtractor
from rie_extract.adapters.vlm_ocr import (
    ScannedNotEnabledExtractor,
    VlmOcrExtractor,
)
from rie_extract.vlm_client import OllamaVlmClient

_LOG = logging.getLogger(__name__)


@runtime_checkable
class Extractor(Protocol):
    """Local Protocol mirroring DocumentExtractorPort.extract signature."""

    def extract(
        self, doc_meta: DocumentMeta, raw: bytes
    ) -> tuple[Sequence[Element], Sequence[StructureEdge]]: ...


@dataclass
class AdapterRouter:
    """Stateful router — carries the VLM enable flag across calls.

    The function-form :func:`pick_adapter` remains the single-shot entry
    point; constructing this dataclass lets the orchestrator wire the
    VLM toggle once at startup rather than per-document.
    """

    enable_vlm: bool = False

    def pick(
        self, doc_meta: DocumentMeta, *, source_path: str | Path | None = None
    ) -> Extractor:
        return pick_adapter(doc_meta, source_path=source_path, enable_vlm=self.enable_vlm)


def pick_adapter(
    doc_meta: DocumentMeta,
    *,
    source_path: str | Path | None = None,
    enable_vlm: bool | None = None,
) -> Extractor:
    """Return the best-fit extractor for ``doc_meta``.

    Routing heuristic:
      • PDF (.pdf / mime/extension hint) → PyMuPDF then Docling (§5.2 chain).
      • HTML (.htm/.html or document URL hint) → BeautifulSoup.
      • Anything else → plain-text extractor (also the .txt fallback).
      • Scanned / VLM-OCR path:
          - if ``OLLAMA_VLM_MODEL`` env is set OR ``enable_vlm=True`` →
            :class:`VlmOcrExtractor` driving :class:`OllamaVlmClient`.
          - else → :class:`ScannedNotEnabledExtractor` (the §11
            graceful-degradation stub that emits an empty placeholder
            element + a stdlib WARNING — it never crashes).
    """
    hint_parts = _hint_parts(doc_meta, source_path)
    hint = " ".join(hint_parts)
    vlm_enabled = _vlm_enabled(enable_vlm)

    if "scanned" in hint or "ocr" in hint:
        if vlm_enabled:
            _LOG.info(
                "Routing doc_id=%s to VlmOcrExtractor (model-generated text; not 'faithful')",
                doc_meta.doc_id,
            )
            return VlmOcrExtractor(vlm_client=OllamaVlmClient.from_env())
        _LOG.warning(
            "Scanned hint detected for doc_id=%s but VLM-OCR is disabled — "
            "falling back to ScannedNotEnabledExtractor (§11 graceful degradation)",
            doc_meta.doc_id,
        )
        return ScannedNotEnabledExtractor()

    if _any_part_suffix(hint_parts, ".pdf"):
        return BornDigitalPdfExtractor()
    if _any_part_suffix(hint_parts, ".html", ".htm") or "<html" in hint:
        return HtmlExtractor()
    if _any_part_suffix(hint_parts, ".txt"):
        return TextExtractor()
    # Default: plain text. Safer than guessing PDF on raw bytes.
    return TextExtractor()


def _vlm_enabled(explicit: bool | None) -> bool:
    """VLM is on iff explicitly requested OR ``OLLAMA_VLM_MODEL`` is set."""
    if explicit is True:
        return True
    if explicit is False:
        # Explicit ``False`` overrides env — useful in tests.
        return False
    return bool(os.environ.get("OLLAMA_VLM_MODEL"))


def _hint_parts(doc_meta: DocumentMeta, source_path: str | Path | None) -> list[str]:
    parts: list[str] = []
    if source_path is not None:
        parts.append(str(source_path).lower())
    if doc_meta.source_url:
        parts.append(doc_meta.source_url.lower())
    parts.append(doc_meta.title.lower())
    return parts


def _any_part_suffix(parts: Sequence[str], *suffixes: str) -> bool:
    return any(part.endswith(suffix) for part in parts for suffix in suffixes)
