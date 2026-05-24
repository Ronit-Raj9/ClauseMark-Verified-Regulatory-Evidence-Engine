"""VLM-OCR extractor for scanned / image-only PDFs.

This is the §5.2 *scanned* path — the model **generates** the page text,
so the resulting elements MUST NOT be labelled "faithful". Per
``CLAUDE.md`` and ``systemArchitecture.md`` §5.2 / §11:

  • Every element is tagged with :class:`OcrEngine.VLM`.
  • ``corrected`` is ``False`` until a human edits the row.
  • Per-page ``extraction_confidence`` is the value returned by the
    injected :class:`VlmClient`. A page below ``OCR_CONFIDENCE_THRESHOLD``
    (``0.7`` by default; override with the ``OCR_CONFIDENCE_THRESHOLD``
    env var) emits a stdlib ``logging.WARNING`` and is *kept* — the
    system never silently drops the page; degradation routes it for
    review (§11).
  • Structure-graph construction is identical to the text path: the
    deterministic legal-numbering parser and
    :class:`StructureGraphBuilder` run over the synthesised elements so
    cross-references resolve the same way.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

import pymupdf  # type: ignore[import-untyped]
from rie_contracts import (
    BoundingBox,
    DocumentMeta,
    Element,
    ElementType,
    OcrEngine,
    StructureEdge,
)

from rie_extract.errors import ExtractionError
from rie_extract.structure.graph_builder import StructureGraphBuilder
from rie_extract.structure.legal_numbering import parse_numbering
from rie_extract.vlm_client import FakeVlmClient, VlmClient

_LOG = logging.getLogger(__name__)

# Default per-page confidence below which we WARN. Env override supported
# so ops can tighten it for high-stakes corpora without a code change.
_DEFAULT_OCR_CONFIDENCE_THRESHOLD = 0.7
_OCR_CONFIDENCE_THRESHOLD_ENV = "OCR_CONFIDENCE_THRESHOLD"

# Render DPI for page → PNG. 200 DPI is the sweet spot for legal text
# (small footnotes still legible) without blowing up the request size.
_RENDER_DPI = 200

_HEADING_RE = re.compile(
    r"^\s*(?:CHAPTER\s+(?:[IVXLCDM]+|\d+)|"
    r"(?:Section|Sec\.?|s\.)\s+\d+[A-Za-z]?|"
    r"(?:Article|Art\.?)\s+\d+[A-Za-z]?)\b",
    re.IGNORECASE,
)


def _ocr_confidence_threshold() -> float:
    """Read the env-configurable confidence threshold, with a safe fallback."""
    raw = os.environ.get(_OCR_CONFIDENCE_THRESHOLD_ENV)
    if not raw:
        return _DEFAULT_OCR_CONFIDENCE_THRESHOLD
    try:
        v = float(raw)
    except ValueError:
        _LOG.warning(
            "Invalid %s=%r — falling back to %.2f",
            _OCR_CONFIDENCE_THRESHOLD_ENV,
            raw,
            _DEFAULT_OCR_CONFIDENCE_THRESHOLD,
        )
        return _DEFAULT_OCR_CONFIDENCE_THRESHOLD
    return max(0.0, min(1.0, v))


@dataclass
class VlmOcrExtractor:
    """Adapter that drives a VLM client over each rendered page.

    The structure-graph build step is run *inside* the adapter (rather
    than relying on :class:`ExtractionService` to do it) so callers who
    use the adapter directly still get cross-reference edges — important
    because the §5.2 scanned path tends to be invoked through the router
    rather than the full service in scripts.
    """

    vlm_client: VlmClient = field(default_factory=FakeVlmClient)
    graph_builder: StructureGraphBuilder = field(default_factory=StructureGraphBuilder)
    render_dpi: int = _RENDER_DPI

    def extract(
        self, doc_meta: DocumentMeta, raw: bytes
    ) -> tuple[Sequence[Element], Sequence[StructureEdge]]:
        _LOG.info(
            "vlm-ocr path engaged for doc_id=%s (model-generated text — NOT 'faithful')",
            doc_meta.doc_id,
        )

        try:
            pdf = pymupdf.open(stream=raw, filetype="pdf")
        except Exception as exc:  # pragma: no cover — pymupdf raises various types
            raise ExtractionError(f"PyMuPDF failed to open scanned PDF: {exc}") from exc

        elements: list[Element] = []
        running_offset = 0
        chapter_id: str | None = None
        section_id: str | None = None
        index = 0
        threshold = _ocr_confidence_threshold()

        try:
            for page_index, page in enumerate(pdf, start=1):
                png_bytes = _render_page_png(page, dpi=self.render_dpi)
                page_text, page_confidence = self.vlm_client.transcribe(png_bytes)

                if page_confidence < threshold:
                    _LOG.warning(
                        "vlm-ocr page %d for doc_id=%s below confidence threshold "
                        "(%.2f < %.2f) — element kept with corrected=False for review",
                        page_index,
                        doc_meta.doc_id,
                        page_confidence,
                        threshold,
                    )

                # Synthesise a page-level bbox at the rendered size — gives
                # the audit viewer something to highlight against the PNG.
                rect = page.rect
                page_bbox = BoundingBox(
                    page=page_index,
                    x0=float(rect.x0),
                    y0=float(rect.y0),
                    x1=float(rect.x1),
                    y1=float(rect.y1),
                )

                # Always emit at least one element per page, even if the
                # VLM returned an empty string — §11 graceful degradation:
                # the document is *known*, the page was *attempted*, the
                # absence is *recorded* with confidence=0.
                paragraphs = _split_paragraphs(page_text)
                if not paragraphs:
                    eid = f"{doc_meta.doc_id}_vlm_p{page_index}_empty"
                    elements.append(
                        Element(
                            element_id=eid,
                            doc_id=doc_meta.doc_id,
                            parent_id=section_id or chapter_id,
                            element_type=ElementType.PARAGRAPH,
                            text="",
                            page=page_index,
                            bbox=page_bbox,
                            char_start=running_offset,
                            char_end=running_offset,
                            extraction_confidence=max(0.0, min(1.0, page_confidence)),
                            ocr_engine=OcrEngine.VLM,
                            corrected=False,
                            legal_numbering=None,
                        )
                    )
                    index += 1
                    continue

                for para in paragraphs:
                    stripped = para.strip()
                    if not stripped:
                        running_offset += len(para) + 2  # paragraph break accounting
                        continue

                    first_line = stripped.splitlines()[0]
                    is_heading = bool(_HEADING_RE.match(first_line))
                    numbering = parse_numbering(first_line) if is_heading else None

                    if is_heading:
                        head_lower = first_line.lower()
                        if "chapter" in head_lower:
                            etype = ElementType.HEADING
                        elif head_lower.startswith(("article", "art.")):
                            etype = ElementType.ARTICLE
                        else:
                            etype = ElementType.SECTION
                    else:
                        etype = ElementType.PARAGRAPH

                    char_start = running_offset
                    char_end = char_start + len(para)

                    eid = (
                        f"{doc_meta.doc_id}_{numbering.replace(' ', '').replace('.', '')}"
                        if numbering
                        else f"{doc_meta.doc_id}_vlm_p{page_index}_{etype.value}_{index}"
                    )
                    index += 1

                    parent_id: str | None
                    if etype == ElementType.HEADING:
                        parent_id = None
                        chapter_id = eid
                        section_id = None
                    elif etype in {ElementType.SECTION, ElementType.ARTICLE}:
                        parent_id = chapter_id
                        section_id = eid
                    else:
                        parent_id = section_id or chapter_id

                    display_text = re.sub(r"\s+", " ", stripped).strip()

                    elements.append(
                        Element(
                            element_id=eid,
                            doc_id=doc_meta.doc_id,
                            parent_id=parent_id,
                            element_type=etype,
                            text=display_text,
                            page=page_index,
                            bbox=page_bbox,
                            char_start=char_start,
                            char_end=char_end,
                            extraction_confidence=max(0.0, min(1.0, page_confidence)),
                            ocr_engine=OcrEngine.VLM,
                            corrected=False,
                            legal_numbering=numbering,
                        )
                    )
                    running_offset = char_end + 2  # double-newline separator
        finally:
            pdf.close()

        # Run the deterministic structure-graph builder over the synthesised
        # element stream. Identical to the text-path post-process so the
        # cross-reference / definition / proviso edges look the same to
        # downstream consumers no matter which extractor produced them.
        edges = list(self.graph_builder.build(elements))
        return elements, edges


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _render_page_png(page: object, *, dpi: int) -> bytes:
    """Render a pymupdf page to PNG bytes — no extra image dep."""
    # pymupdf API: ``page.get_pixmap(dpi=...)`` then ``pix.tobytes("png")``.
    pix = page.get_pixmap(dpi=dpi)  # type: ignore[attr-defined]
    return bytes(pix.tobytes("png"))


def _split_paragraphs(text: str) -> list[str]:
    """Split a page transcription on blank lines (double newline)."""
    if not text:
        return []
    # Tolerate \r\n + extra whitespace runs.
    chunks = re.split(r"\n\s*\n", text.replace("\r\n", "\n"))
    return [c for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# Graceful-degradation stub for §11 — used by the router when VLM is OFF.
# ---------------------------------------------------------------------------


@dataclass
class ScannedNotEnabledExtractor:
    """Returns an explicit one-element absence — never crashes the pipeline.

    This is the §11 graceful-degradation path: a scanned source arrived
    but the operator has not enabled the VLM-OCR backend. We log a
    WARNING, emit a single empty element at confidence=0 with
    ``OcrEngine.NONE`` and ``corrected=False`` so coverage analysis sees
    the document as *attempted but unread*, and return.

    Calling code can detect this case by the element's empty text +
    confidence==0 and route the document to the ``insufficient_coverage``
    bucket rather than emitting a false "no evidence" claim.
    """

    def extract(
        self, doc_meta: DocumentMeta, raw: bytes
    ) -> tuple[Sequence[Element], Sequence[StructureEdge]]:
        del raw  # we deliberately do not read bytes — VLM is the only reader
        _LOG.warning(
            "Scanned document doc_id=%s received but VLM-OCR is disabled "
            "(set OLLAMA_VLM_MODEL or enable_vlm=True to enable). "
            "Emitting empty placeholder element for §11 graceful degradation.",
            doc_meta.doc_id,
        )
        eid = f"{doc_meta.doc_id}_scanned_not_enabled"
        element = Element(
            element_id=eid,
            doc_id=doc_meta.doc_id,
            parent_id=None,
            element_type=ElementType.PARAGRAPH,
            text="",
            page=1,
            bbox=None,
            char_start=0,
            char_end=0,
            extraction_confidence=0.0,
            ocr_engine=OcrEngine.NONE,
            corrected=False,
            legal_numbering=None,
        )
        return [element], []
