"""ProvisionRecord: required-field validation, discovery_tag enum, rationale
truncation, confidence range, indicator display shape, article paragraph warn."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from rie_output.record import CSV_COLUMNS, MAX_RATIONALE_CHARS, ProvisionRecord


def _valid_kwargs(**overrides: str) -> dict[str, str]:
    base = {
        "economy": "Malaysia",
        "law_name": "Personal Data Protection Act 2010",
        "law_number_ref": "Act 709",
        "last_amended": "",
        "indicator_id": "P6-I4",
        "article": "s. 129(1)",
        "discovery_tag": "KNOWN",
        "location_reference": "p. 88",
        "verbatim_snippet": "A data user shall not transfer any personal data...",
        "mapping_rationale": "Cross-border transfer restriction maps to consent regime.",
        "source_url": "https://example.gov.my/act709",
        "confidence": "0.92",
        "notes": "",
    }
    base.update(overrides)
    return base


def test_field_order_matches_csv_columns() -> None:
    order = tuple(ProvisionRecord.model_fields.keys())[: len(CSV_COLUMNS)]
    assert order == CSV_COLUMNS


def test_valid_record_constructs() -> None:
    rec = ProvisionRecord(**_valid_kwargs())
    assert rec.economy == "Malaysia"
    assert rec.as_row()[0] == "Malaysia"
    assert len(rec.as_row()) == 13


@pytest.mark.parametrize(
    "field",
    [
        "economy",
        "law_name",
        "law_number_ref",
        "indicator_id",
        "article",
        "discovery_tag",
        "verbatim_snippet",
        "source_url",
    ],
)
def test_required_field_empty_rejected(field: str) -> None:
    with pytest.raises(ValidationError):
        ProvisionRecord(**_valid_kwargs(**{field: ""}))


@pytest.mark.parametrize("field", ["economy", "verbatim_snippet"])
def test_required_field_whitespace_only_rejected(field: str) -> None:
    with pytest.raises(ValidationError):
        ProvisionRecord(**_valid_kwargs(**{field: "   "}))


def test_last_amended_may_be_blank() -> None:
    rec = ProvisionRecord(**_valid_kwargs(last_amended=""))
    assert rec.last_amended == ""


@pytest.mark.parametrize("tag", ["NEW", "KNOWN"])
def test_discovery_tag_accepts_enum(tag: str) -> None:
    rec = ProvisionRecord(**_valid_kwargs(discovery_tag=tag))
    assert rec.discovery_tag == tag


@pytest.mark.parametrize("tag", ["new", "Known", "DISCOVERED", "", "MAYBE"])
def test_discovery_tag_rejects_non_enum(tag: str) -> None:
    with pytest.raises(ValidationError):
        ProvisionRecord(**_valid_kwargs(discovery_tag=tag))


def test_rationale_truncated_at_300_with_warning() -> None:
    long_rationale = "x" * 450
    rec = ProvisionRecord(**_valid_kwargs(mapping_rationale=long_rationale))
    assert len(rec.mapping_rationale) == MAX_RATIONALE_CHARS
    assert any("truncated" in w for w in rec.warnings)


def test_rationale_at_limit_not_truncated() -> None:
    exact = "y" * MAX_RATIONALE_CHARS
    rec = ProvisionRecord(**_valid_kwargs(mapping_rationale=exact))
    assert len(rec.mapping_rationale) == MAX_RATIONALE_CHARS
    assert not any("truncated" in w for w in rec.warnings)


@pytest.mark.parametrize("conf", ["", "0.00", "0.80", "1.00", "0.92"])
def test_confidence_valid_range(conf: str) -> None:
    rec = ProvisionRecord(**_valid_kwargs(confidence=conf))
    assert rec.confidence == conf


@pytest.mark.parametrize("conf", ["1.50", "2.00", "0.8", "abc", "-0.10", "1"])
def test_confidence_invalid_rejected(conf: str) -> None:
    with pytest.raises(ValidationError):
        ProvisionRecord(**_valid_kwargs(confidence=conf))


@pytest.mark.parametrize("ind", ["6.4", "p6-i4", "P6I4", "Art6", "P6-4", "6-4"])
def test_indicator_id_must_be_display_alias(ind: str) -> None:
    with pytest.raises(ValidationError):
        ProvisionRecord(**_valid_kwargs(indicator_id=ind))


@pytest.mark.parametrize("ind", ["P6-I4", "P7-I2", "P12-I10"])
def test_indicator_id_display_alias_accepted(ind: str) -> None:
    rec = ProvisionRecord(**_valid_kwargs(indicator_id=ind))
    assert rec.indicator_id == ind


def test_article_without_paragraph_warns() -> None:
    rec = ProvisionRecord(**_valid_kwargs(article="Art 26"))
    assert any("paragraph" in w for w in rec.warnings)


def test_article_with_paragraph_no_warn() -> None:
    rec = ProvisionRecord(**_valid_kwargs(article="Art. 26(2)"))
    assert not any("paragraph" in w for w in rec.warnings)
