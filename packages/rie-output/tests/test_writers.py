"""Writers: CSV header/order, verbatim round-trip through csv.reader, JSON
extended fields, validate_rows catches each violation class."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from rie_output.record import CSV_COLUMNS, ProvisionRecord
from rie_output.writers import (
    EXTENDED_PROVISION_FIELDS,
    EvidencePackageJson,
    provision_to_json_dict,
    validate_rows,
    write_csv,
    write_json,
)


def _record(**overrides: str) -> ProvisionRecord:
    base = {
        "economy": "Malaysia",
        "law_name": "Personal Data Protection Act 2010",
        "law_number_ref": "Act 709",
        "last_amended": "",
        "indicator_id": "P6-I4",
        "article": "s. 129(1)",
        "discovery_tag": "KNOWN",
        "location_reference": "p. 88",
        "verbatim_snippet": "A data user shall not transfer personal data.",
        "mapping_rationale": "maps to consent regime",
        "source_url": "https://example.gov.my/act709",
        "confidence": "0.92",
        "notes": "",
    }
    base.update(overrides)
    return ProvisionRecord(**base)


# ── CSV header + column order ───────────────────────────────────────────────
def test_csv_header_exact_columns_in_order(tmp_path: Path) -> None:
    out = write_csv([_record()], tmp_path / "out.csv")
    with out.open(encoding="utf-8", newline="") as fh:
        header = next(csv.reader(fh))
    assert tuple(header) == CSV_COLUMNS
    assert len(header) == 13


def test_csv_one_row_per_provision(tmp_path: Path) -> None:
    out = write_csv([_record(), _record(discovery_tag="NEW")], tmp_path / "out.csv")
    with out.open(encoding="utf-8", newline="") as fh:
        reader = list(csv.reader(fh))
    assert len(reader) == 3  # header + 2 rows


# ── verbatim snippet with commas, quotes, newline round-trips ───────────────
def test_verbatim_with_special_chars_round_trips(tmp_path: Path) -> None:
    nasty = 'Section 1, "transfer", and\nnew-line clause; "quoted, comma".'
    rec = _record(verbatim_snippet=nasty)
    out = write_csv([rec], tmp_path / "out.csv")
    with out.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == list(CSV_COLUMNS)
    snippet_idx = CSV_COLUMNS.index("verbatim_snippet")
    assert rows[1][snippet_idx] == nasty


def test_all_fields_round_trip_by_column_name(tmp_path: Path) -> None:
    rec = _record(verbatim_snippet='has, comma "and" quote')
    out = write_csv([rec], tmp_path / "out.csv")
    with out.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        row = next(reader)
    for col in CSV_COLUMNS:
        assert row[col] == getattr(rec, col)


def test_csv_utf8_non_ascii(tmp_path: Path) -> None:
    rec = _record(economy="Côte d'Ivoire", verbatim_snippet="données à caractère")
    out = write_csv([rec], tmp_path / "out.csv")
    with out.open(encoding="utf-8", newline="") as fh:
        row = next(csv.DictReader(fh))
    assert row["economy"] == "Côte d'Ivoire"
    assert row["verbatim_snippet"] == "données à caractère"


# ── JSON evidence package: extended fields present ──────────────────────────
def test_json_has_extended_fields(tmp_path: Path) -> None:
    prov = provision_to_json_dict(
        _record(),
        source_pdf_path="/data/act709.pdf",
        ocr_quality_cer=0.012,
        processing_time_seconds=4.2,
        pdf_is_scanned=True,
        retrieval_method="hybrid_dense_sparse_rerank",
        raw_context_before="...preceding text...",
        raw_context_after="...following text...",
    )
    for col in CSV_COLUMNS:
        assert col in prov
    for ext in EXTENDED_PROVISION_FIELDS:
        assert ext in prov
    assert prov["source_pdf_path"] == "/data/act709.pdf"
    assert prov["pdf_is_scanned"] is True

    pkg = EvidencePackageJson(
        economy="Malaysia",
        pillar_ids=["P6"],
        generated_at="2026-06-07T00:00:00Z",
        model_version="qwen3-32b@2025-09 + tesseract-5.3",
        provisions=[prov],
        counts={"provisions": 1, "new": 0, "known": 1},
        cost={"ocr_usd": 0.01, "llm_usd": 0.03},
    )
    out = write_json(pkg, tmp_path / "out.json")
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["economy"] == "Malaysia"
    assert loaded["provisions"][0]["retrieval_method"] == "hybrid_dense_sparse_rerank"
    assert loaded["counts"]["known"] == 1
    assert loaded["cost"]["llm_usd"] == 0.03
    for ext in EXTENDED_PROVISION_FIELDS:
        assert ext in loaded["provisions"][0]


# ── validate_rows: empty == conformant, and catches each violation class ────
def test_validate_rows_conformant_empty() -> None:
    assert validate_rows([_record(), _record(discovery_tag="NEW")]) == []


def test_validate_rows_article_without_paragraph() -> None:
    # construct a record that passes (Art with no subsection still constructs,
    # only warns), then validate_rows must flag it.
    rec = _record(article="Art 26")
    violations = validate_rows([rec])
    assert any("paragraph" in v for v in violations)


def test_validate_rows_reports_row_index() -> None:
    bad = _record(article="Art 99")
    violations = validate_rows([_record(), bad])
    assert any(v.startswith("row 2") for v in violations)


def test_validate_rows_long_rationale_not_silently_passed() -> None:
    # ProvisionRecord truncates, so a constructed record is always <= 300.
    rec = _record()
    # simulate an externally-hydrated record whose rationale exceeds the cap.
    object.__setattr__(rec, "mapping_rationale", "z" * 400)
    violations = validate_rows([rec])
    assert any("mapping_rationale" in v for v in violations)


def test_validate_rows_bad_indicator_caught() -> None:
    rec = _record()
    object.__setattr__(rec, "indicator_id", "6.4")
    violations = validate_rows([rec])
    assert any("indicator_id" in v for v in violations)


def test_validate_rows_bad_discovery_tag_caught() -> None:
    rec = _record()
    object.__setattr__(rec, "discovery_tag", "MAYBE")
    violations = validate_rows([rec])
    assert any("discovery_tag" in v for v in violations)


def test_validate_rows_missing_required_caught() -> None:
    rec = _record()
    object.__setattr__(rec, "source_url", "")
    violations = validate_rows([rec])
    assert any("source_url" in v for v in violations)


def test_validate_rows_bad_confidence_caught() -> None:
    rec = _record()
    object.__setattr__(rec, "confidence", "1.50")
    violations = validate_rows([rec])
    assert any("confidence" in v for v in violations)
