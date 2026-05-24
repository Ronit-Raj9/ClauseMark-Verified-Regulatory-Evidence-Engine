"""HTTP-boundary Pydantic schemas.

These wrap or re-export the frozen `rie_contracts` models with HTTP-friendly
shapes (status enums as strings, optional fields nullable, deterministic
example payloads for the OpenAPI doc). The internal models remain the source
of truth; routers convert at the boundary only.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from rie_contracts import (
    Claim,
    ClausePattern,
    CoverageRecord,
    CoverageState,
    Decomposition,
    EvidenceSpan,
    GateName,
    GateResult,
    IndicatorConfig,
    Jurisdiction,
    Layer1Status,
    LegalRegime,
    PillarConfig,
    RegistryEntry,
    ReviewDecision,
    ReviewRecord,
    ScoreBand,
    VerificationReport,
    VerificationStatus,
)

# ════════════════════════════════════════════════════════════════════════════
# Run lifecycle
# ════════════════════════════════════════════════════════════════════════════


class RunRequest(BaseModel):
    """Trigger a pipeline run for one jurisdiction over one or more pillars.

    `run_id` is optional — if omitted the API mints one.
    """

    model_config = ConfigDict(extra="forbid")

    jurisdiction: Jurisdiction
    pillar_ids: list[str] = Field(min_length=1)
    run_id: str | None = None


class RunResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    status: Literal["accepted", "running", "completed", "failed", "skipped"]
    detail: str = ""


class RunStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    jurisdiction: Jurisdiction | None = None
    status: Literal["accepted", "running", "completed", "failed", "skipped", "unknown"]
    claim_count: int = 0
    verified_count: int = 0
    flagged_count: int = 0
    coverage_count: int = 0


# ════════════════════════════════════════════════════════════════════════════
# Claims / verifications (HTTP-friendly DTOs)
# ════════════════════════════════════════════════════════════════════════════


class EvidenceSpanDTO(BaseModel):
    model_config = ConfigDict(extra="forbid")

    span_id: str
    element_id: str
    doc_id: str
    char_start: int
    char_end: int
    role: str

    @classmethod
    def from_model(cls, span: EvidenceSpan) -> EvidenceSpanDTO:
        return cls(
            span_id=span.span_id,
            element_id=span.element_id,
            doc_id=span.doc_id,
            char_start=span.char_start,
            char_end=span.char_end,
            role=span.role.value,
        )


class GateResultDTO(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gate: str
    passed: bool
    detail: str = ""
    score: float | None = None
    ran_at: datetime

    @classmethod
    def from_model(cls, g: GateResult) -> GateResultDTO:
        return cls(
            gate=g.gate.value,
            passed=g.passed,
            detail=g.detail,
            score=g.score,
            ran_at=g.ran_at,
        )


class VerificationReportDTO(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str
    gates: list[GateResultDTO]
    status: str
    failure_reasons: list[str] = Field(default_factory=list)

    @classmethod
    def from_model(cls, report: VerificationReport) -> VerificationReportDTO:
        return cls(
            claim_id=report.claim_id,
            gates=[GateResultDTO.from_model(g) for g in report.gates],
            status=report.status.value,
            failure_reasons=list(report.failure_reasons),
        )


class CitationDTO(BaseModel):
    """A materialised citation — deterministic, NEVER LLM-authored."""

    model_config = ConfigDict(extra="forbid")

    span_id: str
    doc_id: str
    element_id: str
    text: str
    char_start: int
    char_end: int
    role: str


class ClaimDTO(BaseModel):
    """HTTP-friendly Claim. Pattern + status returned as plain strings."""

    model_config = ConfigDict(extra="forbid")

    claim_id: str
    indicator_id: str
    pillar_id: str
    clause_id: str
    jurisdiction: str
    clause_pattern: str
    decomposition: Decomposition
    evidence_spans: list[EvidenceSpanDTO]
    regime: LegalRegime
    layer1_status: str
    model_confidence: float | None = None
    self_consistency_votes: dict[str, int] = Field(default_factory=dict)
    created_at: datetime

    @classmethod
    def from_model(cls, c: Claim) -> ClaimDTO:
        return cls(
            claim_id=c.claim_id,
            indicator_id=c.indicator_id,
            pillar_id=c.pillar_id,
            clause_id=c.clause_id,
            jurisdiction=c.jurisdiction,
            clause_pattern=c.clause_pattern.value,
            decomposition=c.decomposition,
            evidence_spans=[EvidenceSpanDTO.from_model(s) for s in c.evidence_spans],
            regime=c.regime,
            layer1_status=c.layer1_status.value,
            model_confidence=c.model_confidence,
            self_consistency_votes=dict(c.self_consistency_votes),
            created_at=c.created_at,
        )


class ClaimDetailResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim: ClaimDTO
    verification: VerificationReportDTO | None = None
    citations: list[CitationDTO] = Field(default_factory=list)


class ClaimListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[ClaimDTO]
    total: int


# ════════════════════════════════════════════════════════════════════════════
# Coverage
# ════════════════════════════════════════════════════════════════════════════


class CoverageDTO(BaseModel):
    model_config = ConfigDict(extra="forbid")

    jurisdiction: str
    indicator_id: str
    state: str
    measured_recall: float | None = None
    reason: str | None = None
    verified_claim_ids: list[str] = Field(default_factory=list)

    @classmethod
    def from_model(cls, c: CoverageRecord) -> CoverageDTO:
        return cls(
            jurisdiction=c.jurisdiction,
            indicator_id=c.indicator_id,
            state=c.state.value,
            measured_recall=c.measured_recall,
            reason=c.reason,
            verified_claim_ids=list(c.verified_claim_ids),
        )


class CoverageListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[CoverageDTO]
    total: int


# ════════════════════════════════════════════════════════════════════════════
# Reviews
# ════════════════════════════════════════════════════════════════════════════


class ReviewSubmit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str
    reviewer: str = Field(min_length=1, max_length=128)
    decision: ReviewDecision
    corrected_score: ScoreBand | None = None
    note: str = ""


class ReviewDTO(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str
    reviewer: str
    decision: str
    corrected_score: str | None = None
    note: str = ""
    decided_at: datetime

    @classmethod
    def from_model(cls, r: ReviewRecord) -> ReviewDTO:
        return cls(
            claim_id=r.claim_id,
            reviewer=r.reviewer,
            decision=r.decision.value,
            corrected_score=r.corrected_score.value if r.corrected_score else None,
            note=r.note,
            decided_at=r.decided_at,
        )


class ReviewListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[ReviewDTO]
    total: int


# ════════════════════════════════════════════════════════════════════════════
# Audit (the evidence package — used by the audit UI)
# ════════════════════════════════════════════════════════════════════════════


class EvidencePackageClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim: ClaimDTO
    verification: VerificationReportDTO | None = None
    citations: list[CitationDTO] = Field(default_factory=list)


class EvidencePackage(BaseModel):
    """Full audit payload for one jurisdiction — Layer-1 facts + coverage."""

    model_config = ConfigDict(extra="forbid")

    jurisdiction: str
    claims: list[EvidencePackageClaim]
    coverage: list[CoverageDTO]
    generated_at: datetime


# ════════════════════════════════════════════════════════════════════════════
# Pillars
# ════════════════════════════════════════════════════════════════════════════


class PillarSummary(BaseModel):
    """Lightweight pillar row for the registry listing endpoint."""

    model_config = ConfigDict(extra="forbid")

    pillar_id: str
    pillar_name: str
    cluster: str
    document_profile: str
    status: str
    config_path: str
    gold_set_path: str | None = None

    @classmethod
    def from_entry(cls, e: RegistryEntry) -> PillarSummary:
        return cls(
            pillar_id=e.pillar_id,
            pillar_name=e.pillar_name,
            cluster=e.cluster,
            document_profile=e.document_profile.value,
            status=e.status,
            config_path=e.config_path,
            gold_set_path=e.gold_set_path,
        )


class PillarsListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[PillarSummary]
    total: int


class IndicatorDTO(BaseModel):
    model_config = ConfigDict(extra="forbid")

    indicator_id: str
    name: str
    definition: str
    evaluation_method: str
    clause_pattern: str | None = None

    @classmethod
    def from_model(cls, i: IndicatorConfig) -> IndicatorDTO:
        return cls(
            indicator_id=i.indicator_id,
            name=i.name,
            definition=i.definition,
            evaluation_method=i.evaluation_method.value,
            clause_pattern=i.clause_pattern.value if i.clause_pattern else None,
        )


class PillarDetailResponse(BaseModel):
    """Full pillar config — returned to authors / the UI."""

    model_config = ConfigDict(extra="forbid")

    pillar_id: str
    pillar_name: str
    cluster: str
    document_profile: str
    status: str
    description: str = ""
    indicators: list[IndicatorDTO]

    @classmethod
    def from_model(cls, p: PillarConfig) -> PillarDetailResponse:
        return cls(
            pillar_id=p.pillar_id,
            pillar_name=p.pillar_name,
            cluster=p.cluster,
            document_profile=p.document_profile.value,
            status=p.status,
            description=p.description,
            indicators=[IndicatorDTO.from_model(i) for i in p.indicators],
        )


# ════════════════════════════════════════════════════════════════════════════
# Health
# ════════════════════════════════════════════════════════════════════════════


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "degraded", "unhealthy"]
    components: dict[str, str]


__all__: tuple[str, ...] = (
    # Run
    "RunRequest",
    "RunResponse",
    "RunStatusResponse",
    # Claims
    "ClaimDTO",
    "ClaimDetailResponse",
    "ClaimListResponse",
    "EvidenceSpanDTO",
    "GateResultDTO",
    "VerificationReportDTO",
    "CitationDTO",
    # Coverage
    "CoverageDTO",
    "CoverageListResponse",
    # Reviews
    "ReviewSubmit",
    "ReviewDTO",
    "ReviewListResponse",
    # Audit
    "EvidencePackage",
    "EvidencePackageClaim",
    # Pillars
    "PillarSummary",
    "PillarsListResponse",
    "PillarDetailResponse",
    "IndicatorDTO",
    # Health
    "HealthResponse",
    # Re-exported enums (string-coerced at HTTP boundary anyway, but kept for callers)
    "ClausePattern",
    "CoverageState",
    "Layer1Status",
    "ReviewDecision",
    "ScoreBand",
    "VerificationStatus",
    "GateName",
)
