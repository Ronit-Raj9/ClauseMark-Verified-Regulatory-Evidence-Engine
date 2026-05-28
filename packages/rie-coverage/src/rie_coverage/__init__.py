"""3-state absence reasoner.

Implements the §7 coverage contract: the system never emits a bare ``0``.
A ``(jurisdiction, indicator)`` pair lands in exactly one of three states:

* ``EVIDENCE_FOUND``                     — at least one verified claim.
* ``NO_EVIDENCE_IN_SEARCHED_CORPUS``     — nothing verified, but we know the
  gold-set recall, so the miss risk is bounded.
* ``INSUFFICIENT_COVERAGE``              — nothing verified AND no recall has
  been measured for this indicator (or the source was unreachable).

Public surface:

* :class:`CoverageReasoner` — `CoverageReasonerPort` implementation.
* :mod:`rie_coverage.policy` — pure-function helpers documenting the policy.
"""

from rie_coverage.corpus_completeness import (
    DEFAULT_EXPECTED_SOURCE_TYPES,
    DOWNGRADE_THRESHOLD,
    CompletenessReport,
    CorpusManifest,
    EnrichedCoverageRow,
    compute_completeness,
    default_manifest,
    enrich_with_completeness,
    format_corpus_incomplete_reason,
    load_manifest,
)
from rie_coverage.layer2_service import Layer2ScoringService
from rie_coverage.policy import (
    classify,
    format_no_evidence_reason,
    miss_pct_from_recall,
    requires_authoritative_source_check,
)
from rie_coverage.scoring_policy import (
    build_layer2_from_pillar,
    build_layer2_recommendation,
    recommend_band,
)
from rie_coverage.service import CoverageReasoner

__all__ = [
    "DEFAULT_EXPECTED_SOURCE_TYPES",
    "DOWNGRADE_THRESHOLD",
    "CompletenessReport",
    "CorpusManifest",
    "CoverageReasoner",
    "EnrichedCoverageRow",
    "Layer2ScoringService",
    "build_layer2_from_pillar",
    "build_layer2_recommendation",
    "classify",
    "compute_completeness",
    "default_manifest",
    "enrich_with_completeness",
    "format_corpus_incomplete_reason",
    "format_no_evidence_reason",
    "load_manifest",
    "miss_pct_from_recall",
    "recommend_band",
    "requires_authoritative_source_check",
]
