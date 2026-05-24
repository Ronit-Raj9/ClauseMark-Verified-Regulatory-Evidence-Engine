// Plain TypeScript mirrors of the FastAPI schemas in packages/rie-api/.
//
// source: rie-contracts (Pydantic models in packages/rie-contracts/) — keep in
// sync manually for now; codegen (openapi-typescript) is Phase 3 roadmap.
//
// Names mirror schemas.py / rie_contracts enums 1:1. Anything not surfaced
// through the HTTP boundary is intentionally absent.

// ── Enums (string-coerced at the HTTP boundary) ────────────────────────────

export type CoverageState =
  | "evidence_found"
  | "no_evidence_in_searched_corpus"
  | "insufficient_coverage";

export type Layer1Status =
  | "draft"
  | "verified"
  | "flagged"
  | "rejected";

export type VerificationStatus = "verified" | "flagged" | "rejected";

export type ReviewDecision = "accept" | "correct" | "reject";

export type ScoreBand = "0" | "0.5" | "1" | "no_evidence" | "insufficient_coverage";

export type GateName =
  | "span_existence"
  | "verbatim_match"
  | "entailment"
  | "self_consistency";

export type SpanRole = "primary" | "supporting" | "boundary";

export type ClausePattern = string;
export type LegalRegime = string;
export type Jurisdiction = string;

// ── Evidence + verification ────────────────────────────────────────────────

export interface EvidenceSpanDTO {
  span_id: string;
  element_id: string;
  doc_id: string;
  char_start: number;
  char_end: number;
  role: SpanRole | string;
}

export interface GateResultDTO {
  gate: GateName | string;
  passed: boolean;
  detail: string;
  score: number | null;
  ran_at: string; // ISO timestamp
}

export interface VerificationReportDTO {
  claim_id: string;
  gates: GateResultDTO[];
  status: VerificationStatus | string;
  failure_reasons: string[];
}

export interface CitationDTO {
  span_id: string;
  doc_id: string;
  element_id: string;
  text: string;
  char_start: number;
  char_end: number;
  role: string;
}

// Decomposition is open-ended by design (the LLM picks from enums, but the
// shape varies by clause pattern). We keep it as an opaque record so the UI
// renders it generically without hard-coding any indicator.
export type Decomposition = Record<string, unknown>;

export interface ClaimDTO {
  claim_id: string;
  indicator_id: string;
  pillar_id: string;
  clause_id: string;
  jurisdiction: string;
  clause_pattern: string;
  decomposition: Decomposition;
  evidence_spans: EvidenceSpanDTO[];
  regime: LegalRegime;
  layer1_status: Layer1Status | string;
  model_confidence: number | null;
  self_consistency_votes: Record<string, number>;
  created_at: string;
}

export interface ClaimDetailResponse {
  claim: ClaimDTO;
  verification: VerificationReportDTO | null;
  citations: CitationDTO[];
}

export interface ClaimListResponse {
  items: ClaimDTO[];
  total: number;
}

// ── Coverage ───────────────────────────────────────────────────────────────

export interface CoverageDTO {
  jurisdiction: string;
  indicator_id: string;
  state: CoverageState | string;
  measured_recall: number | null;
  reason: string | null;
  verified_claim_ids: string[];
}

export interface CoverageListResponse {
  items: CoverageDTO[];
  total: number;
}

// ── Reviews (HITL) ─────────────────────────────────────────────────────────

export interface ReviewSubmit {
  claim_id: string;
  reviewer: string;
  decision: ReviewDecision;
  corrected_score?: ScoreBand | null;
  note?: string;
}

export interface ReviewDTO {
  claim_id: string;
  reviewer: string;
  decision: string;
  corrected_score: string | null;
  note: string;
  decided_at: string;
}

export interface ReviewListResponse {
  items: ReviewDTO[];
  total: number;
}

// ── Audit package ──────────────────────────────────────────────────────────

export interface EvidencePackageClaim {
  claim: ClaimDTO;
  verification: VerificationReportDTO | null;
  citations: CitationDTO[];
}

export interface EvidencePackage {
  jurisdiction: string;
  claims: EvidencePackageClaim[];
  coverage: CoverageDTO[];
  generated_at: string;
}

// ── Pillars ────────────────────────────────────────────────────────────────

export interface PillarSummary {
  pillar_id: string;
  pillar_name: string;
  cluster: string;
  document_profile: string;
  status: string;
  config_path: string;
  gold_set_path: string | null;
}

export interface PillarsListResponse {
  items: PillarSummary[];
  total: number;
}

export interface IndicatorDTO {
  indicator_id: string;
  name: string;
  definition: string;
  evaluation_method: string;
  clause_pattern: string | null;
}

export interface PillarDetailResponse {
  pillar_id: string;
  pillar_name: string;
  cluster: string;
  document_profile: string;
  status: string;
  description: string;
  indicators: IndicatorDTO[];
}

// ── Runs ───────────────────────────────────────────────────────────────────

export interface RunRequest {
  jurisdiction: string;
  pillar_ids: string[];
  run_id?: string | null;
}

export type RunStatus =
  | "accepted"
  | "running"
  | "completed"
  | "failed"
  | "skipped"
  | "unknown";

export interface RunResponse {
  run_id: string;
  status: RunStatus;
  detail: string;
}

export interface RunStatusResponse {
  run_id: string;
  jurisdiction: string | null;
  status: RunStatus;
  claim_count: number;
  verified_count: number;
  flagged_count: number;
  coverage_count: number;
}

// ── Health ─────────────────────────────────────────────────────────────────

export interface HealthResponse {
  status: "ok" | "degraded" | "unhealthy";
  components: Record<string, string>;
}
