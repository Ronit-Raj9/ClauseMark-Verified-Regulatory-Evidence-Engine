"""Deterministic writers for the conformance output contract.

* :func:`write_csv`   — the primary 13-column CSV (judge-validated).
* :class:`EvidencePackageJson` / :func:`write_json` — the supplementary JSON
  evidence package for the technical judge (13 fields + extended technical
  fields + counts + cost).
* :func:`validate_rows` — pure conformance check returning a list of human
  violation strings (empty list == conformant).

Only stdlib ``csv`` / ``json`` + pydantic + the local record module are used.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from rie_output.indicator_id import is_display_indicator
from rie_output.record import (
    _ARTICLE_HAS_PARAGRAPH_RE,  # reuse the heuristic
    _CONFIDENCE_RE,
    CSV_COLUMNS,
    DISCOVERY_TAGS,
    MAX_RATIONALE_CHARS,
    REQUIRED_FIELDS,
    ProvisionRecord,
)


# ════════════════════════════════════════════════════════════════════════════
# CSV writer — the primary judge-validated artefact.
# ════════════════════════════════════════════════════════════════════════════
def write_csv(rows: list[ProvisionRecord], path: str | Path) -> Path:
    """Write ``rows`` to ``path`` as the 13-column conformance CSV.

    * Header is exactly :data:`CSV_COLUMNS`, in order.
    * One CSV row per provision, fields in column order.
    * Proper RFC-4180 quoting (verbatim snippets routinely contain commas,
      double-quotes and embedded newlines) via :class:`csv.writer` defaults
      (``QUOTE_MINIMAL``, ``\r\n`` line terminator).
    * UTF-8 encoded; ``newline=""`` so the csv module owns line endings.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_COLUMNS)
        for record in rows:
            writer.writerow(record.as_row())
    return out_path


# ════════════════════════════════════════════════════════════════════════════
# JSON evidence package — supplementary, technical judge.
# ════════════════════════════════════════════════════════════════════════════
class EvidencePackageJson(BaseModel):
    """The supplementary JSON package: provisions + technical provenance.

    ``provisions`` carries the 13 contract fields PLUS the extended technical
    fields enumerated in :data:`EXTENDED_PROVISION_FIELDS` (source pdf path,
    OCR quality, timing, scanned flag, retrieval method, raw context windows).
    """

    model_config = ConfigDict(extra="forbid")

    economy: str
    pillar_ids: list[str] = Field(default_factory=list)
    generated_at: str
    model_version: str
    provisions: list[dict[str, object]] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)
    cost: dict[str, float] = Field(default_factory=dict)


#: Extended (non-CSV) technical fields each JSON provision should carry.
EXTENDED_PROVISION_FIELDS: tuple[str, ...] = (
    "source_pdf_path",
    "ocr_quality_cer",
    "processing_time_seconds",
    "pdf_is_scanned",
    "retrieval_method",
    "raw_context_before",
    "raw_context_after",
)


def provision_to_json_dict(
    record: ProvisionRecord,
    *,
    source_pdf_path: str = "",
    ocr_quality_cer: float | None = None,
    processing_time_seconds: float | None = None,
    pdf_is_scanned: bool = False,
    retrieval_method: str = "",
    raw_context_before: str = "",
    raw_context_after: str = "",
) -> dict[str, object]:
    """Materialise one JSON provision: the 13 contract fields + extended fields."""
    base: dict[str, object] = {col: getattr(record, col) for col in CSV_COLUMNS}
    base.update(
        {
            "source_pdf_path": source_pdf_path,
            "ocr_quality_cer": ocr_quality_cer,
            "processing_time_seconds": processing_time_seconds,
            "pdf_is_scanned": pdf_is_scanned,
            "retrieval_method": retrieval_method,
            "raw_context_before": raw_context_before,
            "raw_context_after": raw_context_after,
        }
    )
    return base


def write_json(pkg: EvidencePackageJson, path: str | Path) -> Path:
    """Write the evidence package to ``path`` as UTF-8 pretty JSON."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(pkg.model_dump(), fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    return out_path


# ════════════════════════════════════════════════════════════════════════════
# Conformance validator — empty list == conformant.
# ════════════════════════════════════════════════════════════════════════════
def validate_rows(rows: list[ProvisionRecord]) -> list[str]:
    """Return a list of human-readable conformance violations across ``rows``.

    Checks (per row, prefixed with the 1-based row index):

    * missing required field (:data:`REQUIRED_FIELDS`),
    * bad ``discovery_tag`` (not in {"NEW", "KNOWN"}),
    * ``article`` without a paragraph/subsection,
    * ``mapping_rationale`` exceeding :data:`MAX_RATIONALE_CHARS`,
    * ``confidence`` outside ``""`` / "0.00"-"1.00",
    * ``indicator_id`` not in ``P{n}-I{m}`` display form.

    An empty list means every row is conformant. ``ProvisionRecord`` already
    enforces most of these at construction; this re-validates raw values so the
    same gate works even for records hydrated by other means.
    """
    violations: list[str] = []
    for idx, record in enumerate(rows, start=1):
        for field in REQUIRED_FIELDS:
            if getattr(record, field, "").strip() == "":
                violations.append(f"row {idx}: missing required field {field!r}")

        if record.discovery_tag not in DISCOVERY_TAGS:
            violations.append(
                f"row {idx}: discovery_tag {record.discovery_tag!r} not in {sorted(DISCOVERY_TAGS)}"
            )

        if record.article and not _ARTICLE_HAS_PARAGRAPH_RE.search(record.article):
            violations.append(f"row {idx}: article {record.article!r} lacks a paragraph/subsection")

        if len(record.mapping_rationale) > MAX_RATIONALE_CHARS:
            violations.append(
                f"row {idx}: mapping_rationale is {len(record.mapping_rationale)} chars "
                f"(> {MAX_RATIONALE_CHARS})"
            )

        if record.confidence != "" and not _CONFIDENCE_RE.match(record.confidence):
            violations.append(f"row {idx}: confidence {record.confidence!r} not in '' or 0.00-1.00")
        elif record.confidence != "" and float(record.confidence) > 1.0:
            violations.append(f"row {idx}: confidence {record.confidence!r} exceeds 1.00")

        if not is_display_indicator(record.indicator_id):
            violations.append(
                f"row {idx}: indicator_id {record.indicator_id!r} not in P{{n}}-I{{m}} form"
            )

    return violations
