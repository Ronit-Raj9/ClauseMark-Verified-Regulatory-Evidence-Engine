"""Cross-corpus completeness signals for defensible absence reasoning.

Phase 2 extension to the 3-state reasoner. The base service can already say
"no evidence in the corpus we searched, miss risk ~X% based on gold-set
recall". Reviewers also need a second, orthogonal signal: did we even search
the *kind* of source where this obligation would normally live?

If a jurisdiction's manifest expects a ``tier_2_regulation`` for indicator
6.4 and we only ingested the statute, the absence is NOT defensible without
a regulator search. This module computes that signal and (optionally)
downgrades the coverage state accordingly.

Wiring rules:

* This module imports only :mod:`rie_contracts` enums + standard library +
  ``pyyaml``. No reach-around imports of other adapters.
* The contract ``CoverageRecord`` is frozen. We expose a richer sibling
  :class:`EnrichedCoverageRow` returned from
  :func:`enrich_with_completeness`. The orchestrator can persist either.
* A bare ``0`` is still structurally impossible. The downgrade only re-routes
  between the three legal states.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field
from rie_contracts import (
    CoverageRecord,
    CoverageState,
    DocumentType,
    Jurisdiction,
)

# ─── Defaults ───────────────────────────────────────────────────────────────

#: Fallback when no manifest file exists for a jurisdiction. Mirrors §7's
#: "statute + regulation are the minimum defensible search" baseline.
DEFAULT_EXPECTED_SOURCE_TYPES: tuple[DocumentType, ...] = (
    DocumentType.STATUTE,
    DocumentType.REGULATION,
)

#: Completeness ratio below this triggers a downgrade to INSUFFICIENT_COVERAGE.
DOWNGRADE_THRESHOLD: float = 0.5


# ─── Models ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CorpusManifest:
    """Per-(jurisdiction, indicator) expectation of source types.

    Loaded from ``data/coverage/manifests/<jurisdiction>.yaml``. If the file
    is absent the manifest is synthesised from
    :data:`DEFAULT_EXPECTED_SOURCE_TYPES`.
    """

    jurisdiction: str
    per_indicator: Mapping[str, tuple[DocumentType, ...]]
    default_expected: tuple[DocumentType, ...]

    def expected_for(self, indicator_id: str) -> tuple[DocumentType, ...]:
        """Return the expected source types for an indicator, with fallback."""
        return self.per_indicator.get(indicator_id, self.default_expected)


class CompletenessReport(BaseModel):
    """Reviewer-facing summary of corpus completeness for one (juris, indicator)."""

    model_config = ConfigDict(frozen=True)

    jurisdiction: Jurisdiction
    indicator_id: str
    expected_source_types: list[DocumentType]
    ingested_source_types: list[DocumentType]
    missing_source_types: list[DocumentType]
    completeness_ratio: float = Field(ge=0.0, le=1.0)
    attestation_required: bool

    @property
    def is_defensible(self) -> bool:
        """A no-evidence verdict is defensible only when nothing is missing."""
        return not self.missing_source_types


class EnrichedCoverageRow(BaseModel):
    """Richer sibling of :class:`CoverageRecord`.

    Used internally by the coverage package — the contract
    :class:`CoverageRecord` is frozen so we cannot add new fields to it.
    The orchestrator can persist this row alongside the contract record, or
    drop it back to a :class:`CoverageRecord` via :meth:`to_contract_row`.
    """

    model_config = ConfigDict(frozen=True)

    jurisdiction: Jurisdiction
    indicator_id: str
    state: CoverageState
    measured_recall: float | None = Field(default=None, ge=0.0, le=1.0)
    reason: str | None = None
    verified_claim_ids: list[str] = Field(default_factory=list)
    corpus_completeness: CompletenessReport | None = None
    #: Populated by the eval pipeline (see eval-agent). Holds *external*
    #: recall measured against a corpus broader than what we ingested —
    #: e.g., regulator-website recall. ``None`` means "not yet measured".
    external_recall: float | None = Field(default=None, ge=0.0, le=1.0)

    def to_contract_row(self) -> CoverageRecord:
        """Project back to the frozen :class:`CoverageRecord` contract."""
        return CoverageRecord(
            jurisdiction=self.jurisdiction,
            indicator_id=self.indicator_id,
            state=self.state,
            measured_recall=self.measured_recall,
            reason=self.reason,
            verified_claim_ids=list(self.verified_claim_ids),
        )


# ─── Manifest loading ──────────────────────────────────────────────────────


def _manifest_path(repo_root: Path, jurisdiction: str) -> Path:
    return repo_root / "data" / "coverage" / "manifests" / f"{jurisdiction.lower()}.yaml"


def _coerce_doc_types(values: Iterable[Any], where: str) -> tuple[DocumentType, ...]:
    """Coerce a YAML list of strings into ``DocumentType`` enum members.

    Fails LOUD on unknown values — silent dropping would hide manifest typos
    and silently shrink the "expected" set, which would inflate the
    completeness ratio. Better to crash now than to mislead a reviewer.
    """
    out: list[DocumentType] = []
    for raw in values:
        if not isinstance(raw, str):
            msg = f"{where}: expected string document_type, got {type(raw).__name__}"
            raise ValueError(msg)
        try:
            out.append(DocumentType(raw))
        except ValueError as exc:
            msg = f"{where}: unknown document_type '{raw}'"
            raise ValueError(msg) from exc
    return tuple(out)


def default_manifest(jurisdiction: str) -> CorpusManifest:
    """Build the default manifest used when no on-disk file exists."""
    return CorpusManifest(
        jurisdiction=jurisdiction,
        per_indicator={},
        default_expected=DEFAULT_EXPECTED_SOURCE_TYPES,
    )


def load_manifest(repo_root: Path, jurisdiction: str) -> CorpusManifest:
    """Load the manifest for a jurisdiction, falling back to defaults.

    The file is optional. When absent we return :func:`default_manifest`
    so the rest of the pipeline can keep running — the completeness signal
    just becomes "statute + regulation expected".
    """
    path = _manifest_path(repo_root, jurisdiction)
    if not path.exists():
        return default_manifest(jurisdiction)

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        msg = f"{path}: top-level YAML must be a mapping"
        raise ValueError(msg)
    data = cast(dict[str, Any], raw)

    declared = data.get("jurisdiction")
    if declared is not None and str(declared) != jurisdiction:
        msg = (
            f"{path}: manifest declares jurisdiction={declared!r} "
            f"but loader was asked for {jurisdiction!r}"
        )
        raise ValueError(msg)

    default_raw = data.get("default_expected_source_types")
    default_expected: tuple[DocumentType, ...] = (
        _coerce_doc_types(default_raw, f"{path}:default_expected_source_types")
        if isinstance(default_raw, list)
        else DEFAULT_EXPECTED_SOURCE_TYPES
    )

    per_indicator: dict[str, tuple[DocumentType, ...]] = {}
    indicators_raw = data.get("indicators")
    if isinstance(indicators_raw, dict):
        for ind_id, body in cast(dict[str, Any], indicators_raw).items():
            if not isinstance(body, dict):
                msg = f"{path}:indicators.{ind_id}: must be a mapping"
                raise ValueError(msg)
            body_d = cast(dict[str, Any], body)
            expected = body_d.get("expected_source_types")
            if not isinstance(expected, list):
                msg = f"{path}:indicators.{ind_id}.expected_source_types: must be a list"
                raise ValueError(msg)
            per_indicator[str(ind_id)] = _coerce_doc_types(
                expected, f"{path}:indicators.{ind_id}.expected_source_types"
            )

    return CorpusManifest(
        jurisdiction=jurisdiction,
        per_indicator=per_indicator,
        default_expected=default_expected,
    )


# ─── Pure-function completeness logic ───────────────────────────────────────


def compute_completeness(
    jurisdiction: str,
    indicator_id: str,
    ingested_source_types: Sequence[DocumentType],
    manifest: CorpusManifest,
) -> CompletenessReport:
    """Compare ingested source types to the manifest's expectations.

    The completeness ratio is ``|expected & ingested| / |expected|``. When
    no source types are expected (a deliberately empty manifest entry) the
    ratio is 1.0 — there is nothing to be incomplete about.
    """
    expected = manifest.expected_for(indicator_id)
    expected_set = set(expected)
    # De-dup ingested types while preserving deterministic order for the report.
    seen: set[DocumentType] = set()
    ingested_dedup: list[DocumentType] = []
    for t in ingested_source_types:
        if t not in seen:
            seen.add(t)
            ingested_dedup.append(t)

    covered = expected_set & seen
    missing = [t for t in expected if t not in covered]
    ratio = 1.0 if not expected_set else len(covered) / len(expected_set)
    return CompletenessReport(
        jurisdiction=jurisdiction,
        indicator_id=indicator_id,
        expected_source_types=list(expected),
        ingested_source_types=ingested_dedup,
        missing_source_types=missing,
        completeness_ratio=ratio,
        attestation_required=bool(missing),
    )


def format_corpus_incomplete_reason(missing: Sequence[DocumentType]) -> str:
    """Reason string when we downgrade NO_EVIDENCE → INSUFFICIENT_COVERAGE."""
    names = ", ".join(t.value for t in missing) if missing else "<none>"
    return f"corpus incomplete: missing {names}"


def enrich_with_completeness(
    base: CoverageRecord,
    *,
    ingested_source_types: Sequence[DocumentType],
    manifest: CorpusManifest,
    external_recall: float | None = None,
    downgrade_threshold: float = DOWNGRADE_THRESHOLD,
) -> EnrichedCoverageRow:
    """Attach completeness + optionally downgrade the coverage state.

    Rules:

    * Always attach a :class:`CompletenessReport` so reviewers see the
      manifest comparison regardless of state.
    * Only ``NO_EVIDENCE_IN_SEARCHED_CORPUS`` is downgrade-eligible. If the
      completeness ratio is below ``downgrade_threshold`` we re-route the row
      to ``INSUFFICIENT_COVERAGE`` with a reason naming the missing types.
      ``EVIDENCE_FOUND`` is never downgraded — we already have evidence.
      ``INSUFFICIENT_COVERAGE`` stays put (it is already the safe state).
    * The three-state enum is never expanded. Bare ``0`` remains impossible.
    """
    report = compute_completeness(
        jurisdiction=base.jurisdiction,
        indicator_id=base.indicator_id,
        ingested_source_types=ingested_source_types,
        manifest=manifest,
    )

    state = base.state
    reason = base.reason
    measured_recall = base.measured_recall

    if (
        base.state == CoverageState.NO_EVIDENCE_IN_SEARCHED_CORPUS
        and report.completeness_ratio < downgrade_threshold
    ):
        state = CoverageState.INSUFFICIENT_COVERAGE
        reason = format_corpus_incomplete_reason(report.missing_source_types)
        # Drop measured_recall: it bounded miss risk *within* the searched
        # corpus, which we are now declaring too narrow to trust.
        measured_recall = None

    return EnrichedCoverageRow(
        jurisdiction=base.jurisdiction,
        indicator_id=base.indicator_id,
        state=state,
        measured_recall=measured_recall,
        reason=reason,
        verified_claim_ids=list(base.verified_claim_ids),
        corpus_completeness=report,
        external_recall=external_recall,
    )
