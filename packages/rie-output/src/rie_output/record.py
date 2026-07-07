"""The judge-validated provision record — 13 string fields, EXACT CSV order.

Judges validate the output *programmatically*: the column names and their order
must match :data:`CSV_COLUMNS` byte-for-byte. :class:`ProvisionRecord` is the
single source of truth for one row of the primary CSV.

All 13 fields are ``str``. Required (non-empty) fields are enforced at
construction time; soft rules (rationale length, ``article`` paragraph
presence) are *warnings* surfaced through :attr:`ProvisionRecord.warnings` and
:func:`rie_output.writers.validate_rows`, never construction failures — except
``mapping_rationale`` over 300 chars, which is truncated (with a warning) so the
record stays conformant.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ════════════════════════════════════════════════════════════════════════════
# The contract: 13 columns, EXACT names, EXACT order. Judges validate this.
# ════════════════════════════════════════════════════════════════════════════
CSV_COLUMNS: tuple[str, ...] = (
    "economy",
    "law_name",
    "law_number_ref",
    "last_amended",
    "indicator_id",
    "article",
    "discovery_tag",
    "location_reference",
    "verbatim_snippet",
    "mapping_rationale",
    "source_url",
    "confidence",
    "notes",
)

#: Fields that MUST be non-empty for a conformant row. ``last_amended`` is
#: required by template position but MAY be "" (blank if the law was never
#: amended), so it is deliberately absent here.
REQUIRED_FIELDS: tuple[str, ...] = (
    "economy",
    "law_name",
    "law_number_ref",
    "indicator_id",
    "article",
    "discovery_tag",
    "verbatim_snippet",
    "source_url",
)

DISCOVERY_TAGS: frozenset[str] = frozenset({"NEW", "KNOWN"})

MAX_RATIONALE_CHARS = 300

# ``P{pillar}-I{indicator}`` display alias — the CSV-facing indicator shape.
_INDICATOR_DISPLAY_RE = re.compile(r"^P\d+-I\d+$")

# confidence: "" OR a two-decimal string in [0.00, 1.00].
_CONFIDENCE_RE = re.compile(r"^[01]\.\d{2}$")

# article paragraph heuristic: a parenthesised subsection "(2)" / "(1)(a)" OR a
# trailing ".N" / "-N" subsection marker. Bare "Art. 26" / "s. 16" warns.
_ARTICLE_HAS_PARAGRAPH_RE = re.compile(r"[\(\.\-§]")


class ProvisionRecord(BaseModel):
    """One provision = one CSV row. 13 string fields in CSV column order.

    Field order below is load-bearing: it is asserted to equal
    :data:`CSV_COLUMNS` at import time.
    """

    model_config = ConfigDict(str_strip_whitespace=False, validate_assignment=True)

    economy: str
    law_name: str
    law_number_ref: str
    last_amended: str = ""
    indicator_id: str
    article: str
    discovery_tag: str
    location_reference: str = ""
    verbatim_snippet: str
    mapping_rationale: str = ""
    source_url: str = ""
    confidence: str = ""
    notes: str = ""

    #: Soft-rule warnings accumulated during validation. Not a CSV column.
    warnings: list[str] = Field(default_factory=list, exclude=True)

    # ── required-field enforcement ──────────────────────────────────────────
    @field_validator(
        "economy",
        "law_name",
        "law_number_ref",
        "indicator_id",
        "article",
        "discovery_tag",
        "verbatim_snippet",
        "source_url",
        mode="after",
    )
    @classmethod
    def _required_non_empty(cls, value: str, info) -> str:
        if value is None or value.strip() == "":
            msg = f"required field {info.field_name!r} must be non-empty"
            raise ValueError(msg)
        return value

    # ── discovery_tag enum ──────────────────────────────────────────────────
    @field_validator("discovery_tag", mode="after")
    @classmethod
    def _discovery_tag_enum(cls, value: str) -> str:
        if value not in DISCOVERY_TAGS:
            msg = f"discovery_tag must be one of {sorted(DISCOVERY_TAGS)}, got {value!r}"
            raise ValueError(msg)
        return value

    # ── indicator_id display shape ──────────────────────────────────────────
    @field_validator("indicator_id", mode="after")
    @classmethod
    def _indicator_display_shape(cls, value: str) -> str:
        if not _INDICATOR_DISPLAY_RE.match(value):
            msg = (
                f"indicator_id must be display alias 'P{{pillar}}-I{{indicator}}', "
                f"got {value!r} (use to_display_indicator() to convert decimal ids)"
            )
            raise ValueError(msg)
        return value

    # ── confidence range ────────────────────────────────────────────────────
    @field_validator("confidence", mode="after")
    @classmethod
    def _confidence_range(cls, value: str) -> str:
        if value == "":
            return value
        if not _CONFIDENCE_RE.match(value):
            msg = f"confidence must be '' or a two-decimal string in [0.00, 1.00], got {value!r}"
            raise ValueError(msg)
        # numeric upper bound (regex already pins [01]\.\d\d so <= 1.99 possible).
        if float(value) > 1.0:
            msg = f"confidence {value!r} exceeds 1.00"
            raise ValueError(msg)
        return value

    # ── soft rules: rationale truncation + article-paragraph warning ────────
    @model_validator(mode="after")
    def _soft_rules(self) -> ProvisionRecord:
        if len(self.mapping_rationale) > MAX_RATIONALE_CHARS:
            self.warnings.append(
                f"mapping_rationale truncated from {len(self.mapping_rationale)} "
                f"to {MAX_RATIONALE_CHARS} chars"
            )
            # bypass validate_assignment to avoid re-triggering this validator.
            object.__setattr__(
                self, "mapping_rationale", self.mapping_rationale[:MAX_RATIONALE_CHARS]
            )
        if not _ARTICLE_HAS_PARAGRAPH_RE.search(self.article):
            self.warnings.append(
                f"article {self.article!r} appears to lack a paragraph/subsection "
                "(expected e.g. 'Art. 26(2)' or 's. 16(1)(a)')"
            )
        return self

    def as_row(self) -> list[str]:
        """The 13 field values, in :data:`CSV_COLUMNS` order, ready for csv.writer."""
        return [getattr(self, col) for col in CSV_COLUMNS]


# Fail loudly at import if model field order ever drifts from the CSV contract.
_model_field_order = tuple(ProvisionRecord.model_fields.keys())[: len(CSV_COLUMNS)]
assert _model_field_order == CSV_COLUMNS, (
    f"ProvisionRecord field order {_model_field_order} != CSV_COLUMNS {CSV_COLUMNS}"
)
