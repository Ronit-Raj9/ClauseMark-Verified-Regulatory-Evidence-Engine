"""Tests for cross-corpus completeness signals.

Covers:

* manifest loading from YAML
* default-manifest fallback when no file exists
* completeness ratio math (perfect / partial / missing / empty-expected)
* enrich_with_completeness downgrade rules (only NO_EVIDENCE is eligible)
* the three-state enum is never exceeded — bare ``0`` impossible
* the external_recall field is plumbed end-to-end
"""

from __future__ import annotations

from pathlib import Path

import pytest
from rie_contracts import CoverageRecord, CoverageState, DocumentType
from rie_coverage import (
    DEFAULT_EXPECTED_SOURCE_TYPES,
    CompletenessReport,
    CorpusManifest,
    EnrichedCoverageRow,
    compute_completeness,
    default_manifest,
    enrich_with_completeness,
    format_corpus_incomplete_reason,
    load_manifest,
)


# ─── load_manifest ──────────────────────────────────────────────────────────


class TestLoadManifest:
    def test_missing_file_returns_default_manifest(self, tmp_path: Path) -> None:
        manifest = load_manifest(tmp_path, "NOWHERE")
        assert isinstance(manifest, CorpusManifest)
        assert manifest.jurisdiction == "NOWHERE"
        assert manifest.per_indicator == {}
        assert manifest.default_expected == DEFAULT_EXPECTED_SOURCE_TYPES

    def test_loads_per_indicator_and_default(self, tmp_path: Path) -> None:
        path = tmp_path / "data" / "coverage" / "manifests" / "sample.yaml"
        path.parent.mkdir(parents=True)
        path.write_text(
            "jurisdiction: SAMPLE\n"
            "indicators:\n"
            "  '6.4':\n"
            "    expected_source_types: [statute, regulation, guideline]\n"
            "default_expected_source_types: [statute, regulation]\n",
            encoding="utf-8",
        )
        manifest = load_manifest(tmp_path, "SAMPLE")
        assert manifest.jurisdiction == "SAMPLE"
        assert manifest.expected_for("6.4") == (
            DocumentType.STATUTE,
            DocumentType.REGULATION,
            DocumentType.GUIDELINE,
        )
        # Fallback to default for unknown indicator.
        assert manifest.expected_for("99.9") == (
            DocumentType.STATUTE,
            DocumentType.REGULATION,
        )

    def test_lowercases_filename(self, tmp_path: Path) -> None:
        """``load_manifest("SAMPLE")`` reads ``sample.yaml`` (lowercase)."""
        path = tmp_path / "data" / "coverage" / "manifests" / "sample.yaml"
        path.parent.mkdir(parents=True)
        path.write_text(
            "jurisdiction: SAMPLE\nindicators: {}\n"
            "default_expected_source_types: [statute]\n",
            encoding="utf-8",
        )
        manifest = load_manifest(tmp_path, "SAMPLE")
        assert manifest.default_expected == (DocumentType.STATUTE,)

    def test_rejects_unknown_document_type(self, tmp_path: Path) -> None:
        path = tmp_path / "data" / "coverage" / "manifests" / "sample.yaml"
        path.parent.mkdir(parents=True)
        path.write_text(
            "jurisdiction: SAMPLE\n"
            "indicators:\n"
            "  '6.4':\n"
            "    expected_source_types: [statute, made_up_type]\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="unknown document_type"):
            load_manifest(tmp_path, "SAMPLE")

    def test_rejects_mismatched_jurisdiction_in_file(self, tmp_path: Path) -> None:
        path = tmp_path / "data" / "coverage" / "manifests" / "sample.yaml"
        path.parent.mkdir(parents=True)
        path.write_text(
            "jurisdiction: OTHER\nindicators: {}\n", encoding="utf-8"
        )
        with pytest.raises(ValueError, match="declares jurisdiction"):
            load_manifest(tmp_path, "SAMPLE")

    def test_rejects_non_mapping_top_level(self, tmp_path: Path) -> None:
        path = tmp_path / "data" / "coverage" / "manifests" / "sample.yaml"
        path.parent.mkdir(parents=True)
        path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
        with pytest.raises(ValueError, match="must be a mapping"):
            load_manifest(tmp_path, "SAMPLE")


# ─── default_manifest ───────────────────────────────────────────────────────


def test_default_manifest_uses_statute_plus_regulation() -> None:
    manifest = default_manifest("ZZ")
    assert manifest.default_expected == (
        DocumentType.STATUTE,
        DocumentType.REGULATION,
    )
    assert manifest.expected_for("anything") == (
        DocumentType.STATUTE,
        DocumentType.REGULATION,
    )


# ─── compute_completeness ───────────────────────────────────────────────────


class TestComputeCompleteness:
    def _manifest(
        self, expected: tuple[DocumentType, ...] = DEFAULT_EXPECTED_SOURCE_TYPES
    ) -> CorpusManifest:
        return CorpusManifest(
            jurisdiction="SAMPLE",
            per_indicator={"6.4": expected},
            default_expected=expected,
        )

    def test_perfect_coverage_yields_ratio_one(self) -> None:
        report = compute_completeness(
            "SAMPLE",
            "6.4",
            [DocumentType.STATUTE, DocumentType.REGULATION],
            self._manifest(),
        )
        assert report.completeness_ratio == 1.0
        assert report.missing_source_types == []
        assert report.attestation_required is False
        assert report.is_defensible is True

    def test_half_coverage_yields_half_ratio_and_missing(self) -> None:
        report = compute_completeness(
            "SAMPLE",
            "6.4",
            [DocumentType.STATUTE],
            self._manifest(),
        )
        assert report.completeness_ratio == 0.5
        assert report.missing_source_types == [DocumentType.REGULATION]
        assert report.attestation_required is True
        assert report.is_defensible is False

    def test_extra_ingested_types_are_ignored_for_ratio(self) -> None:
        report = compute_completeness(
            "SAMPLE",
            "6.4",
            [DocumentType.STATUTE, DocumentType.REGULATION, DocumentType.NOTICE],
            self._manifest(),
        )
        assert report.completeness_ratio == 1.0
        assert report.ingested_source_types == [
            DocumentType.STATUTE,
            DocumentType.REGULATION,
            DocumentType.NOTICE,
        ]

    def test_duplicates_are_deduplicated_in_report(self) -> None:
        report = compute_completeness(
            "SAMPLE",
            "6.4",
            [DocumentType.STATUTE, DocumentType.STATUTE, DocumentType.STATUTE],
            self._manifest(),
        )
        assert report.ingested_source_types == [DocumentType.STATUTE]
        assert report.missing_source_types == [DocumentType.REGULATION]

    def test_nothing_expected_yields_ratio_one(self) -> None:
        manifest = CorpusManifest(
            jurisdiction="SAMPLE",
            per_indicator={"6.4": ()},
            default_expected=(),
        )
        report = compute_completeness("SAMPLE", "6.4", [], manifest)
        assert report.completeness_ratio == 1.0
        assert report.missing_source_types == []
        assert report.attestation_required is False

    def test_zero_ingested_with_expectations_is_zero_ratio(self) -> None:
        report = compute_completeness("SAMPLE", "6.4", [], self._manifest())
        assert report.completeness_ratio == 0.0
        assert report.attestation_required is True
        assert set(report.missing_source_types) == {
            DocumentType.STATUTE,
            DocumentType.REGULATION,
        }


# ─── format_corpus_incomplete_reason ────────────────────────────────────────


def test_reason_mentions_missing_types() -> None:
    msg = format_corpus_incomplete_reason(
        [DocumentType.REGULATION, DocumentType.GUIDELINE]
    )
    assert "corpus incomplete" in msg
    assert "regulation" in msg
    assert "guideline" in msg


# ─── enrich_with_completeness ───────────────────────────────────────────────


def _base_record(state: CoverageState, recall: float | None = None) -> CoverageRecord:
    return CoverageRecord(
        jurisdiction="SAMPLE",
        indicator_id="6.4",
        state=state,
        measured_recall=recall,
        reason=None,
        verified_claim_ids=[],
    )


class TestEnrichWithCompleteness:
    def _manifest(self) -> CorpusManifest:
        return CorpusManifest(
            jurisdiction="SAMPLE",
            per_indicator={
                "6.4": (DocumentType.STATUTE, DocumentType.REGULATION),
            },
            default_expected=(DocumentType.STATUTE, DocumentType.REGULATION),
        )

    def test_no_evidence_with_full_coverage_is_not_downgraded(self) -> None:
        row = enrich_with_completeness(
            _base_record(CoverageState.NO_EVIDENCE_IN_SEARCHED_CORPUS, 0.82),
            ingested_source_types=[DocumentType.STATUTE, DocumentType.REGULATION],
            manifest=self._manifest(),
        )
        assert isinstance(row, EnrichedCoverageRow)
        assert row.state == CoverageState.NO_EVIDENCE_IN_SEARCHED_CORPUS
        assert row.measured_recall == 0.82
        assert row.corpus_completeness is not None
        assert row.corpus_completeness.completeness_ratio == 1.0
        assert row.corpus_completeness.attestation_required is False

    def test_no_evidence_below_threshold_is_downgraded(self) -> None:
        row = enrich_with_completeness(
            _base_record(CoverageState.NO_EVIDENCE_IN_SEARCHED_CORPUS, 0.82),
            ingested_source_types=[],  # 0/2 = 0.0 < 0.5
            manifest=self._manifest(),
        )
        assert row.state == CoverageState.INSUFFICIENT_COVERAGE
        assert row.reason is not None and "corpus incomplete" in row.reason
        # measured_recall is dropped because the bound applied to a too-narrow corpus.
        assert row.measured_recall is None
        assert row.corpus_completeness is not None
        assert row.corpus_completeness.completeness_ratio == 0.0

    def test_no_evidence_exactly_at_threshold_is_not_downgraded(self) -> None:
        """Threshold is strict-less-than. ratio == 0.5 with default keeps the state."""
        row = enrich_with_completeness(
            _base_record(CoverageState.NO_EVIDENCE_IN_SEARCHED_CORPUS, 0.7),
            ingested_source_types=[DocumentType.STATUTE],  # 1/2 = 0.5
            manifest=self._manifest(),
        )
        assert row.state == CoverageState.NO_EVIDENCE_IN_SEARCHED_CORPUS
        assert row.corpus_completeness is not None
        assert row.corpus_completeness.completeness_ratio == 0.5
        assert row.corpus_completeness.attestation_required is True

    def test_custom_threshold_can_raise_the_bar(self) -> None:
        row = enrich_with_completeness(
            _base_record(CoverageState.NO_EVIDENCE_IN_SEARCHED_CORPUS, 0.7),
            ingested_source_types=[DocumentType.STATUTE],  # ratio 0.5
            manifest=self._manifest(),
            downgrade_threshold=0.75,
        )
        assert row.state == CoverageState.INSUFFICIENT_COVERAGE

    def test_evidence_found_never_downgraded(self) -> None:
        base = CoverageRecord(
            jurisdiction="SAMPLE",
            indicator_id="6.4",
            state=CoverageState.EVIDENCE_FOUND,
            measured_recall=0.82,
            reason=None,
            verified_claim_ids=["c1"],
        )
        row = enrich_with_completeness(
            base,
            ingested_source_types=[],  # ratio 0 — but we have evidence
            manifest=self._manifest(),
        )
        assert row.state == CoverageState.EVIDENCE_FOUND
        assert row.verified_claim_ids == ["c1"]
        # Completeness still attached for reviewer transparency.
        assert row.corpus_completeness is not None
        assert row.corpus_completeness.completeness_ratio == 0.0

    def test_insufficient_coverage_stays_insufficient(self) -> None:
        row = enrich_with_completeness(
            _base_record(CoverageState.INSUFFICIENT_COVERAGE, None),
            ingested_source_types=[DocumentType.STATUTE, DocumentType.REGULATION],
            manifest=self._manifest(),
        )
        assert row.state == CoverageState.INSUFFICIENT_COVERAGE
        # Even with full coverage, we don't upgrade — that would silently
        # invent evidence we never observed.
        assert row.corpus_completeness is not None
        assert row.corpus_completeness.completeness_ratio == 1.0

    def test_external_recall_is_plumbed_through(self) -> None:
        row = enrich_with_completeness(
            _base_record(CoverageState.NO_EVIDENCE_IN_SEARCHED_CORPUS, 0.82),
            ingested_source_types=[DocumentType.STATUTE, DocumentType.REGULATION],
            manifest=self._manifest(),
            external_recall=0.61,
        )
        assert row.external_recall == 0.61

    def test_external_recall_defaults_to_none(self) -> None:
        row = enrich_with_completeness(
            _base_record(CoverageState.NO_EVIDENCE_IN_SEARCHED_CORPUS, 0.82),
            ingested_source_types=[DocumentType.STATUTE, DocumentType.REGULATION],
            manifest=self._manifest(),
        )
        assert row.external_recall is None

    def test_enriched_projects_back_to_contract_row(self) -> None:
        row = enrich_with_completeness(
            _base_record(CoverageState.NO_EVIDENCE_IN_SEARCHED_CORPUS, 0.82),
            ingested_source_types=[DocumentType.STATUTE, DocumentType.REGULATION],
            manifest=self._manifest(),
            external_recall=0.7,
        )
        contract = row.to_contract_row()
        assert isinstance(contract, CoverageRecord)
        assert contract.state == CoverageState.NO_EVIDENCE_IN_SEARCHED_CORPUS
        assert contract.measured_recall == 0.82
        assert contract.verified_claim_ids == []


# ─── Three-state invariant after enrichment ─────────────────────────────────


class TestThreeStateInvariantAfterEnrichment:
    def test_every_combination_stays_in_three_states(self) -> None:
        manifest = CorpusManifest(
            jurisdiction="SAMPLE",
            per_indicator={"6.4": (DocumentType.STATUTE, DocumentType.REGULATION)},
            default_expected=(DocumentType.STATUTE, DocumentType.REGULATION),
        )
        legal = {
            CoverageState.EVIDENCE_FOUND,
            CoverageState.NO_EVIDENCE_IN_SEARCHED_CORPUS,
            CoverageState.INSUFFICIENT_COVERAGE,
        }
        for start in legal:
            for ingested in (
                [],
                [DocumentType.STATUTE],
                [DocumentType.STATUTE, DocumentType.REGULATION],
            ):
                base = CoverageRecord(
                    jurisdiction="SAMPLE",
                    indicator_id="6.4",
                    state=start,
                    measured_recall=0.9 if start != CoverageState.INSUFFICIENT_COVERAGE else None,
                    verified_claim_ids=["c1"] if start == CoverageState.EVIDENCE_FOUND else [],
                )
                row = enrich_with_completeness(
                    base, ingested_source_types=ingested, manifest=manifest
                )
                assert row.state in legal


# ─── CompletenessReport is_defensible ──────────────────────────────────────


def test_is_defensible_true_when_nothing_missing() -> None:
    report = CompletenessReport(
        jurisdiction="SAMPLE",
        indicator_id="6.4",
        expected_source_types=[DocumentType.STATUTE],
        ingested_source_types=[DocumentType.STATUTE],
        missing_source_types=[],
        completeness_ratio=1.0,
        attestation_required=False,
    )
    assert report.is_defensible is True


def test_is_defensible_false_when_anything_missing() -> None:
    report = CompletenessReport(
        jurisdiction="SAMPLE",
        indicator_id="6.4",
        expected_source_types=[DocumentType.STATUTE, DocumentType.REGULATION],
        ingested_source_types=[DocumentType.STATUTE],
        missing_source_types=[DocumentType.REGULATION],
        completeness_ratio=0.5,
        attestation_required=True,
    )
    assert report.is_defensible is False
