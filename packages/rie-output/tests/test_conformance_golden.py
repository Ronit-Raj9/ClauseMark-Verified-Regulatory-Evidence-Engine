"""Golden conformance fixture: Malaysia PDPA s.129 -> CSV -> re-read.

Mirrors the worked-example schema: economy 'Malaysia', law 'Personal Data
Protection Act 2010', law_number_ref 'Act 709', indicator P6-I4, article
's. 129(1)', discovery_tag KNOWN. Write + reread the CSV and assert exact
columns and values survive the round-trip byte-for-byte.
"""

from __future__ import annotations

import csv
from pathlib import Path

from rie_output.indicator_id import to_decimal, to_display_indicator
from rie_output.record import CSV_COLUMNS, ProvisionRecord
from rie_output.writers import validate_rows, write_csv

MALAYSIA_PDPA_S129 = {
    "economy": "Malaysia",
    "law_name": "Personal Data Protection Act 2010",
    "law_number_ref": "Act 709",
    "last_amended": "",
    "indicator_id": "P6-I4",
    "article": "s. 129(1)",
    "discovery_tag": "KNOWN",
    "location_reference": "p. 88",
    "verbatim_snippet": (
        "A data user shall not transfer any personal data of a data subject to a "
        "place outside Malaysia unless to such place as specified by the Minister."
    ),
    "mapping_rationale": (
        "Conditional cross-border transfer restriction maps to consent/adequacy regime (P6-I4)."
    ),
    "source_url": "https://www.kpkt.gov.my/pdpa/act709",
    "confidence": "0.93",
    "notes": "Cross-references s. 130 exceptions.",
}


def test_malaysia_pdpa_s129_round_trip(tmp_path: Path) -> None:
    rec = ProvisionRecord(**MALAYSIA_PDPA_S129)
    assert validate_rows([rec]) == []

    out = write_csv([rec], tmp_path / "Malaysia_P6.csv")

    with out.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))

    # exact 13 columns, exact order.
    assert rows[0] == list(CSV_COLUMNS)
    assert len(rows[0]) == 13

    # values survive verbatim.
    with out.open(encoding="utf-8", newline="") as fh:
        record = next(csv.DictReader(fh))
    for col in CSV_COLUMNS:
        assert record[col] == MALAYSIA_PDPA_S129[col], col


def test_indicator_alias_consistency() -> None:
    # P6-I4 in the CSV corresponds to decimal 6.4 in the gold DB.
    assert to_decimal(MALAYSIA_PDPA_S129["indicator_id"]) == "6.4"
    assert to_display_indicator("6.4") == MALAYSIA_PDPA_S129["indicator_id"]


def test_golden_csv_has_exactly_one_data_row(tmp_path: Path) -> None:
    rec = ProvisionRecord(**MALAYSIA_PDPA_S129)
    out = write_csv([rec], tmp_path / "out.csv")
    with out.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    assert len(rows) == 2  # header + 1 provision
