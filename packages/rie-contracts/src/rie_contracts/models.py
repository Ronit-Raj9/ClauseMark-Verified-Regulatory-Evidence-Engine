"""Pydantic models — the data contract.

Stable identifiers (`doc_id`, `element_id`, `clause_id`, `span_id`) flow through
the pipeline and never change shape. The LLM may emit `indicator_id` (from an
enum-constrained choice) and `span_id` references; everything else is materialised
by deterministic code from these models.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ════════════════════════════════════════════════════════════════════════════
# Enums — every controlled vocabulary lives here.
# ════════════════════════════════════════════════════════════════════════════


class DocumentType(StrEnum):
    STATUTE = "statute"
    AMENDMENT = "amendment"
    REGULATION = "regulation"
    GUIDELINE = "guideline"
    TREATY = "treaty"
    NOTICE = "notice"
    OTHER = "other"


class AuthorityTier(StrEnum):
    TIER_1_STATUTE = "tier_1_statute"
    TIER_2_REGULATION = "tier_2_regulation"
    TIER_3_GUIDELINE = "tier_3_guideline"
    TIER_4_INFORMAL = "tier_4_informal"


class DocumentProfile(StrEnum):
    STATUTORY_LEGAL_TEXT = "statutory_legal_text"
    STRUCTURED_TABULAR = "structured_tabular"
    MIXED_REGULATORY = "mixed_regulatory"
    TREATY_MEMBERSHIP = "treaty_membership"


class EvaluationMethod(StrEnum):
    CLAUSE_EXTRACTION = "clause_extraction"
    TABULAR_LOOKUP = "tabular_lookup"
    TREATY_LOOKUP = "treaty_lookup"


class ClausePattern(StrEnum):
    OBLIGATION = "obligation"
    PROHIBITION = "prohibition"
    CONDITIONAL_REGIME = "conditional_regime"
    EXEMPTION = "exemption"
    DEFINITION = "definition"


class ElementType(StrEnum):
    HEADING = "heading"
    ARTICLE = "article"
    SECTION = "section"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"
    TABLE_CELL = "table_cell"
    FOOTNOTE = "footnote"
    DEFINITION = "definition"
    SCHEDULE = "schedule"
    OTHER = "other"


class StructureEdgeType(StrEnum):
    CROSS_REFERENCE = "cross_reference"
    DEFINES = "defines"
    PROVISO = "proviso"
    AMENDS = "amends"
    NOTWITHSTANDING = "notwithstanding"


class OcrEngine(StrEnum):
    NONE = "none"
    TESSERACT = "tesseract"
    VLM = "vlm"


class GateName(StrEnum):
    SPAN_EXISTENCE = "span_existence"
    VERBATIM_MATCH = "verbatim_match"
    ENTAILMENT = "entailment"
    SELF_CONSISTENCY = "self_consistency"


class VerificationStatus(StrEnum):
    VERIFIED = "verified"
    FLAGGED = "flagged"
    REJECTED = "rejected"


class Layer1Status(StrEnum):
    PENDING_VERIFICATION = "pending_verification"
    VERIFIED = "verified"
    FLAGGED = "flagged"
    REJECTED = "rejected"


class CoverageState(StrEnum):
    EVIDENCE_FOUND = "evidence_found"
    NO_EVIDENCE_IN_SEARCHED_CORPUS = "no_evidence_in_searched_corpus"
    INSUFFICIENT_COVERAGE = "insufficient_coverage"


class ScoreBand(StrEnum):
    ZERO = "0"
    HALF = "0.5"
    ONE = "1"
    NULL = "null"


class ReviewDecision(StrEnum):
    ACCEPT = "accept"
    CORRECT = "correct"
    REJECT = "reject"


class SpanRole(StrEnum):
    PRIMARY = "primary"
    REGIME_MEMBER = "regime_member"
    DEFINITION = "definition"
    EXCEPTION = "exception"


# Free-form but bounded.
Jurisdiction = Annotated[str, Field(min_length=2, max_length=64)]


# ════════════════════════════════════════════════════════════════════════════
# Core models
# ════════════════════════════════════════════════════════════════════════════


class BoundingBox(BaseModel):
    model_config = ConfigDict(frozen=True)
    page: int = Field(ge=1)
    x0: float
    y0: float
    x1: float
    y1: float


class DocumentMeta(BaseModel):
    """Provenance + addressability for a source document."""

    doc_id: str = Field(min_length=1)
    jurisdiction: Jurisdiction
    title: str
    document_type: DocumentType
    effective_date: datetime | None = None
    authority_tier: AuthorityTier
    source_url: str | None = None
    sha256: str = Field(min_length=64, max_length=64)
    retrieved_at: datetime
    language: str = "en"


class Element(BaseModel):
    """An addressable unit of an extracted document."""

    element_id: str
    doc_id: str
    parent_id: str | None = None
    element_type: ElementType
    text: str
    page: int = Field(ge=1)
    bbox: BoundingBox | None = None
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    extraction_confidence: float = Field(ge=0.0, le=1.0)
    ocr_engine: OcrEngine = OcrEngine.NONE
    corrected: bool = False
    legal_numbering: str | None = None

    @model_validator(mode="after")
    def _validate_offsets(self) -> Element:
        if self.char_end < self.char_start:
            msg = f"char_end {self.char_end} < char_start {self.char_start}"
            raise ValueError(msg)
        return self


class StructureEdge(BaseModel):
    model_config = ConfigDict(frozen=True)
    from_element: str
    to_element: str
    edge_type: StructureEdgeType
    raw_reference: str | None = None


# ════════════════════════════════════════════════════════════════════════════
# Retrieval
# ════════════════════════════════════════════════════════════════════════════


class RetrievalHit(BaseModel):
    """A child-chunk hit, with parent element + neighbourhood for whole-law reasoning."""

    element_id: str
    doc_id: str
    score: float
    parent_element_id: str
    neighbourhood_element_ids: list[str] = Field(default_factory=list)
    snippet: str
    dense_score: float | None = None
    sparse_score: float | None = None
    rerank_score: float | None = None


# ════════════════════════════════════════════════════════════════════════════
# Classification + Layer 1
# ════════════════════════════════════════════════════════════════════════════


class Decomposition(BaseModel):
    """Compliance-to-Code style clause decomposition."""

    subject: str
    condition: str | None = None
    constraint: str
    context: str | None = None


class EvidenceSpan(BaseModel):
    """A span reference (NEVER a model-authored citation). Format: `<doc_id>#<start>-<end>`."""

    span_id: str = Field(pattern=r"^[\w\-.]+#\d+-\d+$")
    element_id: str
    doc_id: str
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    role: SpanRole = SpanRole.PRIMARY

    @model_validator(mode="after")
    def _check(self) -> EvidenceSpan:
        expected = f"{self.doc_id}#{self.char_start}-{self.char_end}"
        if self.span_id != expected:
            msg = f"span_id {self.span_id} != derived {expected}"
            raise ValueError(msg)
        if self.char_end < self.char_start:
            raise ValueError("char_end < char_start")
        return self


class LegalRegime(BaseModel):
    """Regime members assembled from the structure graph for whole-law reasoning."""

    primary_element_id: str
    member_element_ids: list[str]
    definitions: list[str] = Field(default_factory=list)
    exceptions: list[str] = Field(default_factory=list)


class Claim(BaseModel):
    """The Layer-1 output: a structured assertion about which indicator a clause maps to."""

    claim_id: str
    indicator_id: str
    pillar_id: str
    clause_id: str
    jurisdiction: Jurisdiction
    clause_pattern: ClausePattern
    decomposition: Decomposition
    evidence_spans: list[EvidenceSpan]
    regime: LegalRegime
    layer1_status: Layer1Status = Layer1Status.PENDING_VERIFICATION
    model_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    self_consistency_votes: dict[str, int] = Field(default_factory=dict)
    created_at: datetime


# ════════════════════════════════════════════════════════════════════════════
# Verification (the 4 gates)
# ════════════════════════════════════════════════════════════════════════════


class GateResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    gate: GateName
    passed: bool
    detail: str = ""
    score: float | None = None
    ran_at: datetime


class VerificationReport(BaseModel):
    """Outcome of running all 4 gates on a claim. Status is derived from gate results."""

    claim_id: str
    gates: list[GateResult]
    status: VerificationStatus
    failure_reasons: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_status(self) -> VerificationReport:
        names = {g.gate for g in self.gates}
        required = {
            GateName.SPAN_EXISTENCE,
            GateName.VERBATIM_MATCH,
            GateName.ENTAILMENT,
            GateName.SELF_CONSISTENCY,
        }
        if not required.issubset(names):
            missing = required - names
            raise ValueError(f"missing gate results: {missing}")
        return self


# ════════════════════════════════════════════════════════════════════════════
# Layer 2 — recommendation (NEVER a final score)
# ════════════════════════════════════════════════════════════════════════════


class Layer2Recommendation(BaseModel):
    """A score recommendation — the system never emits a final score."""

    claim_id: str
    indicator_id: str
    recommended_band: ScoreBand
    rationale: str
    open_questions: list[str] = Field(default_factory=list)
    human_confirmation_required: Literal[True] = True


# ════════════════════════════════════════════════════════════════════════════
# Coverage / absence (3-state, never bare 0)
# ════════════════════════════════════════════════════════════════════════════


class CoverageRecord(BaseModel):
    jurisdiction: Jurisdiction
    indicator_id: str
    state: CoverageState
    measured_recall: float | None = Field(default=None, ge=0.0, le=1.0)
    reason: str | None = None
    verified_claim_ids: list[str] = Field(default_factory=list)


# ════════════════════════════════════════════════════════════════════════════
# Review (the audit trail)
# ════════════════════════════════════════════════════════════════════════════


class ReviewRecord(BaseModel):
    claim_id: str
    reviewer: str
    decision: ReviewDecision
    corrected_score: ScoreBand | None = None
    note: str = ""
    decided_at: datetime


# ════════════════════════════════════════════════════════════════════════════
# Config — pillar / source / gold
# ════════════════════════════════════════════════════════════════════════════


class IndicatorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    indicator_id: str
    name: str
    definition: str
    evaluation_method: EvaluationMethod
    clause_pattern: ClausePattern | None = None
    scoring_criteria: dict[str, str] = Field(default_factory=dict)
    positive_keywords: dict[str, list[str]] = Field(default_factory=dict)
    negative_cues: dict[str, list[str]] = Field(default_factory=dict)
    few_shot_examples: list[dict[str, str | float | dict[str, str]]] = Field(default_factory=list)
    authority_hints: list[AuthorityTier] = Field(default_factory=list)


class PillarConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pillar_id: str
    pillar_name: str
    cluster: str
    document_profile: DocumentProfile
    status: Literal["built", "stub", "roadmap"]
    description: str = ""
    indicators: list[IndicatorConfig]


class RegistryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pillar_id: str
    pillar_name: str
    cluster: str
    document_profile: DocumentProfile
    status: Literal["built", "stub", "roadmap"]
    config_path: str
    gold_set_path: str | None = None


class SourceRegistryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_id: str
    jurisdiction: Jurisdiction
    title: str
    source_url: str | None = None
    document_type: DocumentType
    authority_tier: AuthorityTier
    effective_date: datetime | None = None
    sha256_hash: str | None = None
    language: str = "en"
    local_path: str | None = None


class GoldItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    gold_id: str
    pillar_id: str
    indicator_id: str
    jurisdiction: Jurisdiction
    doc_id: str
    span_text: str
    expected_clause_pattern: ClausePattern
    expected_score_band: ScoreBand
    expected_authority_tier: AuthorityTier
    notes: str = ""
