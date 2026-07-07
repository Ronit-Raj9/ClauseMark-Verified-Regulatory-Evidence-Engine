"""Conformance output writer for the RDTII judge-validated 13-column contract.

Public surface:

* :class:`ProvisionRecord`   — one CSV row; 13 string fields in exact order.
* :data:`CSV_COLUMNS`        — the 13 column names, exact order. Judges validate.
* :func:`write_csv`          — primary 13-column CSV writer.
* :func:`write_json`         — supplementary JSON evidence package writer.
* :class:`EvidencePackageJson` — JSON package model (13 fields + technical).
* :func:`validate_rows`      — conformance gate; empty list == conformant.
* :func:`to_display_indicator` / :func:`to_decimal` — indicator-id aliasing.

Imports only ``rie_contracts`` + stdlib + pydantic.
"""

from rie_output.indicator_id import (
    is_display_indicator,
    to_decimal,
    to_display_indicator,
)
from rie_output.record import (
    CSV_COLUMNS,
    DISCOVERY_TAGS,
    MAX_RATIONALE_CHARS,
    REQUIRED_FIELDS,
    ProvisionRecord,
)
from rie_output.writers import (
    EXTENDED_PROVISION_FIELDS,
    EvidencePackageJson,
    provision_to_json_dict,
    validate_rows,
    write_csv,
    write_json,
)

__all__ = [
    "CSV_COLUMNS",
    "DISCOVERY_TAGS",
    "EXTENDED_PROVISION_FIELDS",
    "MAX_RATIONALE_CHARS",
    "REQUIRED_FIELDS",
    "EvidencePackageJson",
    "ProvisionRecord",
    "is_display_indicator",
    "provision_to_json_dict",
    "to_decimal",
    "to_display_indicator",
    "validate_rows",
    "write_csv",
    "write_json",
]
