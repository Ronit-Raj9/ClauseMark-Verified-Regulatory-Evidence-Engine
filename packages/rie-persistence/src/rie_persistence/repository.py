"""Concrete `DocumentRepositoryPort` over SQLAlchemy + Postgres."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from rie_contracts import (
    AuthorityTier,
    BoundingBox,
    Claim,
    ClausePattern,
    CoverageRecord,
    CoverageState,
    Decomposition,
    DocumentMeta,
    DocumentType,
    Element,
    ElementType,
    EvidenceSpan,
    GateName,
    GateResult,
    Layer1Status,
    Layer2Recommendation,
    LegalRegime,
    OcrEngine,
    ReviewDecision,
    ReviewRecord,
    ScoreBand,
    SpanRole,
    StructureEdge,
    StructureEdgeType,
    VerificationReport,
    VerificationStatus,
)
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from rie_persistence.db import session_scope
from rie_persistence.models import (
    ClaimRow,
    CoverageRow,
    DocumentRow,
    ElementRow,
    EvidenceSpanRow,
    ReviewRow,
    StructureEdgeRow,
    VerificationRow,
)


def _derive_status_inline(gates: list[GateResult]) -> tuple[VerificationStatus, list[str]]:
    """Mirror of rie_domain.derive_status — duplicated to keep persistence pure-adapter."""
    by = {g.gate: g for g in gates}
    failures: list[str] = []
    for det in (GateName.SPAN_EXISTENCE, GateName.VERBATIM_MATCH):
        if det not in by:
            return VerificationStatus.REJECTED, [f"missing gate {det}"]
        if not by[det].passed:
            failures.append(f"{det}: {by[det].detail}")
    if failures:
        return VerificationStatus.REJECTED, failures
    for mod in (GateName.ENTAILMENT, GateName.SELF_CONSISTENCY):
        if mod not in by:
            return VerificationStatus.FLAGGED, [f"missing gate {mod}"]
        if not by[mod].passed:
            failures.append(f"{mod}: {by[mod].detail}")
    if failures:
        return VerificationStatus.FLAGGED, failures
    return VerificationStatus.VERIFIED, []


@dataclass
class DocumentRepository:
    """Postgres-backed implementation of `DocumentRepositoryPort`."""

    session_factory: sessionmaker[Session]

    # ─── Documents + elements + edges ─────────────────────────────────────

    def save_document(self, meta: DocumentMeta) -> None:
        with session_scope(self.session_factory) as s:
            existing = s.get(DocumentRow, meta.doc_id)
            if existing:
                existing.title = meta.title
                existing.jurisdiction = meta.jurisdiction
                existing.document_type = meta.document_type.value
                existing.effective_date = meta.effective_date
                existing.authority_tier = meta.authority_tier.value
                existing.source_url = meta.source_url
                existing.sha256 = meta.sha256
                existing.retrieved_at = meta.retrieved_at
                existing.language = meta.language
            else:
                s.add(
                    DocumentRow(
                        doc_id=meta.doc_id,
                        jurisdiction=meta.jurisdiction,
                        title=meta.title,
                        document_type=meta.document_type.value,
                        effective_date=meta.effective_date,
                        authority_tier=meta.authority_tier.value,
                        source_url=meta.source_url,
                        sha256=meta.sha256,
                        retrieved_at=meta.retrieved_at,
                        language=meta.language,
                    )
                )

    def save_elements(
        self, doc_id: str, elements: Sequence[Element], edges: Sequence[StructureEdge]
    ) -> None:
        with session_scope(self.session_factory) as s:
            for e in elements:
                row = s.get(ElementRow, e.element_id)
                payload = {
                    "doc_id": e.doc_id,
                    "parent_id": e.parent_id,
                    "element_type": e.element_type.value,
                    "text": e.text,
                    "page": e.page,
                    "bbox": e.bbox.model_dump() if e.bbox else None,
                    "char_start": e.char_start,
                    "char_end": e.char_end,
                    "extraction_confidence": e.extraction_confidence,
                    "ocr_engine": e.ocr_engine.value,
                    "corrected": e.corrected,
                    "legal_numbering": e.legal_numbering,
                }
                if row:
                    for k, v in payload.items():
                        setattr(row, k, v)
                else:
                    s.add(ElementRow(element_id=e.element_id, **payload))
            # Dedupe (from, to, type) within this batch.
            seen_keys: set[tuple[str, str, str]] = set()
            edges_to_insert: list[StructureEdge] = []
            for edge in edges:
                key = (edge.from_element, edge.to_element, edge.edge_type.value)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                edges_to_insert.append(edge)
            if edges_to_insert:
                # Pre-fetch existing keys in DB to skip them too (idempotent reruns).
                existing = set(
                    s.execute(
                        select(
                            StructureEdgeRow.from_element,
                            StructureEdgeRow.to_element,
                            StructureEdgeRow.edge_type,
                        ).where(
                            StructureEdgeRow.from_element.in_(
                                {e.from_element for e in edges_to_insert}
                            )
                        )
                    ).all()
                )
                for edge in edges_to_insert:
                    key = (edge.from_element, edge.to_element, edge.edge_type.value)
                    if key in existing:
                        continue
                    s.add(
                        StructureEdgeRow(
                            from_element=edge.from_element,
                            to_element=edge.to_element,
                            edge_type=edge.edge_type.value,
                            raw_reference=edge.raw_reference,
                        )
                    )

    def get_element_text(self, element_id: str) -> str:
        with session_scope(self.session_factory) as s:
            row = s.get(ElementRow, element_id)
            if row is None:
                raise KeyError(element_id)
            return row.text

    def get_element(self, element_id: str) -> Element:
        with session_scope(self.session_factory) as s:
            row = s.get(ElementRow, element_id)
            if row is None:
                raise KeyError(element_id)
            return _to_element(row)

    def get_elements(self, element_ids: Sequence[str]) -> Sequence[Element]:
        if not element_ids:
            return []
        with session_scope(self.session_factory) as s:
            rows = s.scalars(select(ElementRow).where(ElementRow.element_id.in_(element_ids))).all()
            return [_to_element(r) for r in rows]

    def get_structure_edges(self, doc_id: str) -> Sequence[StructureEdge]:
        with session_scope(self.session_factory) as s:
            elem_ids = s.scalars(
                select(ElementRow.element_id).where(ElementRow.doc_id == doc_id)
            ).all()
            if not elem_ids:
                return []
            rows = s.scalars(
                select(StructureEdgeRow).where(StructureEdgeRow.from_element.in_(elem_ids))
            ).all()
            return [
                StructureEdge(
                    from_element=r.from_element,
                    to_element=r.to_element,
                    edge_type=StructureEdgeType(r.edge_type),
                    raw_reference=r.raw_reference,
                )
                for r in rows
            ]

    def get_document(self, doc_id: str) -> DocumentMeta:
        with session_scope(self.session_factory) as s:
            row = s.get(DocumentRow, doc_id)
            if row is None:
                raise KeyError(doc_id)
            return _to_doc_meta(row)

    # ─── Claims + verifications ───────────────────────────────────────────

    def save_claim(self, claim: Claim) -> None:
        with session_scope(self.session_factory) as s:
            existing = s.get(ClaimRow, claim.claim_id)
            payload = {
                "indicator_id": claim.indicator_id,
                "pillar_id": claim.pillar_id,
                "clause_id": claim.clause_id,
                "jurisdiction": claim.jurisdiction,
                "clause_pattern": claim.clause_pattern.value,
                "decomposition": claim.decomposition.model_dump(),
                "regime": claim.regime.model_dump(),
                "layer1_status": claim.layer1_status.value,
                "model_confidence": claim.model_confidence,
                "self_consistency_votes": claim.self_consistency_votes,
                "created_at": claim.created_at,
            }
            if existing:
                for k, v in payload.items():
                    setattr(existing, k, v)
                for old in list(existing.spans):
                    s.delete(old)
                s.flush()
                row = existing
            else:
                row = ClaimRow(claim_id=claim.claim_id, **payload)
                s.add(row)
            for span in claim.evidence_spans:
                s.add(
                    EvidenceSpanRow(
                        span_id=span.span_id,
                        claim_id=claim.claim_id,
                        element_id=span.element_id,
                        doc_id=span.doc_id,
                        char_start=span.char_start,
                        char_end=span.char_end,
                        role=span.role.value,
                    )
                )

    def get_claim(self, claim_id: str) -> Claim:
        with session_scope(self.session_factory) as s:
            row = s.get(ClaimRow, claim_id)
            if row is None:
                raise KeyError(claim_id)
            spans = list(row.spans)
            return _to_claim(row, spans)

    def save_verification(self, report: VerificationReport) -> None:
        with session_scope(self.session_factory) as s:
            for g in report.gates:
                s.add(
                    VerificationRow(
                        claim_id=report.claim_id,
                        gate=g.gate.value,
                        passed=g.passed,
                        detail=g.detail,
                        score=g.score,
                        ran_at=g.ran_at,
                    )
                )
            claim_row = s.get(ClaimRow, report.claim_id)
            if claim_row:
                claim_row.layer1_status = (
                    Layer1Status.VERIFIED.value
                    if report.status == VerificationStatus.VERIFIED
                    else Layer1Status.FLAGGED.value
                    if report.status == VerificationStatus.FLAGGED
                    else Layer1Status.REJECTED.value
                )

    def save_layer2_recommendation(self, recommendation: Layer2Recommendation) -> None:
        """Persist a Layer-2 recommendation on the claim row."""
        with session_scope(self.session_factory) as s:
            row = s.get(ClaimRow, recommendation.claim_id)
            if row is None:
                raise KeyError(recommendation.claim_id)
            row.layer2_recommendation = recommendation.model_dump(mode="json")

    def get_layer2_recommendation(self, claim_id: str) -> Layer2Recommendation | None:
        """Load a persisted Layer-2 recommendation, if any."""
        with session_scope(self.session_factory) as s:
            row = s.get(ClaimRow, claim_id)
            if row is None:
                raise KeyError(claim_id)
            return _to_layer2(row)

    def list_layer2_recommendations(
        self,
        jurisdiction: str | None = None,
        pillar_id: str | None = None,
    ) -> Sequence[Layer2Recommendation]:
        """List persisted Layer-2 recommendations scoped to a run's jurisdiction/pillars."""
        with session_scope(self.session_factory) as s:
            stmt = select(ClaimRow).where(ClaimRow.layer2_recommendation.is_not(None))
            if jurisdiction:
                stmt = stmt.where(ClaimRow.jurisdiction == jurisdiction)
            if pillar_id:
                stmt = stmt.where(ClaimRow.pillar_id == pillar_id)
            rows = s.scalars(stmt).all()
            return [rec for r in rows if (rec := _to_layer2(r)) is not None]

    def latest_verification(self, claim_id: str) -> VerificationReport | None:
        with session_scope(self.session_factory) as s:
            rows = s.scalars(
                select(VerificationRow)
                .where(VerificationRow.claim_id == claim_id)
                .order_by(VerificationRow.ran_at.desc())
            ).all()
            if not rows:
                return None
            seen: dict[GateName, GateResult] = {}
            for r in rows:
                gate = GateName(r.gate)
                if gate not in seen:
                    seen[gate] = GateResult(
                        gate=gate,
                        passed=r.passed,
                        detail=r.detail or "",
                        score=r.score,
                        ran_at=r.ran_at,
                    )
            gates = list(seen.values())
            if len(gates) < 4:
                return None
            status, reasons = _derive_status_inline(gates)
            return VerificationReport(
                claim_id=claim_id, gates=gates, status=status, failure_reasons=reasons
            )

    # ─── Coverage + reviews ───────────────────────────────────────────────

    def save_coverage(self, record: CoverageRecord) -> None:
        with session_scope(self.session_factory) as s:
            existing = s.scalar(
                select(CoverageRow).where(
                    CoverageRow.jurisdiction == record.jurisdiction,
                    CoverageRow.indicator_id == record.indicator_id,
                )
            )
            if existing:
                existing.state = record.state.value
                existing.measured_recall = record.measured_recall
                existing.reason = record.reason
                existing.verified_claim_ids = list(record.verified_claim_ids)
            else:
                s.add(
                    CoverageRow(
                        jurisdiction=record.jurisdiction,
                        indicator_id=record.indicator_id,
                        state=record.state.value,
                        measured_recall=record.measured_recall,
                        reason=record.reason,
                        verified_claim_ids=list(record.verified_claim_ids),
                    )
                )

    def list_coverage(self, jurisdiction: str | None = None) -> Sequence[CoverageRecord]:
        with session_scope(self.session_factory) as s:
            stmt = select(CoverageRow)
            if jurisdiction:
                stmt = stmt.where(CoverageRow.jurisdiction == jurisdiction)
            rows = s.scalars(stmt).all()
            return [_to_coverage(r) for r in rows]

    def save_review(self, review: ReviewRecord) -> None:
        with session_scope(self.session_factory) as s:
            s.add(
                ReviewRow(
                    claim_id=review.claim_id,
                    reviewer=review.reviewer,
                    decision=review.decision.value,
                    corrected_score=(
                        review.corrected_score.value if review.corrected_score else None
                    ),
                    note=review.note,
                    decided_at=review.decided_at,
                )
            )

    def list_claims_for_review(
        self, jurisdiction: str | None = None, pillar_id: str | None = None
    ) -> Sequence[Claim]:
        with session_scope(self.session_factory) as s:
            stmt = select(ClaimRow).where(ClaimRow.layer1_status == Layer1Status.FLAGGED.value)
            if jurisdiction:
                stmt = stmt.where(ClaimRow.jurisdiction == jurisdiction)
            if pillar_id:
                stmt = stmt.where(ClaimRow.pillar_id == pillar_id)
            rows = s.scalars(stmt).all()
            out: list[Claim] = []
            for r in rows:
                spans = list(r.spans)
                out.append(_to_claim(r, spans))
            return out

    def list_reviews(self, claim_id: str) -> Sequence[ReviewRecord]:
        with session_scope(self.session_factory) as s:
            rows = s.scalars(select(ReviewRow).where(ReviewRow.claim_id == claim_id)).all()
            return [
                ReviewRecord(
                    claim_id=r.claim_id,
                    reviewer=r.reviewer,
                    decision=ReviewDecision(r.decision),
                    corrected_score=ScoreBand(r.corrected_score) if r.corrected_score else None,
                    note=r.note,
                    decided_at=r.decided_at,
                )
                for r in rows
            ]


def _to_doc_meta(row: DocumentRow) -> DocumentMeta:
    return DocumentMeta(
        doc_id=row.doc_id,
        jurisdiction=row.jurisdiction,
        title=row.title,
        document_type=DocumentType(row.document_type),
        effective_date=row.effective_date,
        authority_tier=AuthorityTier(row.authority_tier),
        source_url=row.source_url,
        sha256=row.sha256,
        retrieved_at=row.retrieved_at,
        language=row.language,
    )


def _to_element(row: ElementRow) -> Element:
    return Element(
        element_id=row.element_id,
        doc_id=row.doc_id,
        parent_id=row.parent_id,
        element_type=ElementType(row.element_type),
        text=row.text,
        page=row.page,
        bbox=BoundingBox.model_validate(row.bbox) if row.bbox else None,
        char_start=row.char_start,
        char_end=row.char_end,
        extraction_confidence=row.extraction_confidence,
        ocr_engine=OcrEngine(row.ocr_engine),
        corrected=row.corrected,
        legal_numbering=row.legal_numbering,
    )


def _to_claim(row: ClaimRow, spans: list[EvidenceSpanRow]) -> Claim:
    return Claim(
        claim_id=row.claim_id,
        indicator_id=row.indicator_id,
        pillar_id=row.pillar_id,
        clause_id=row.clause_id,
        jurisdiction=row.jurisdiction,
        clause_pattern=ClausePattern(row.clause_pattern),
        decomposition=Decomposition.model_validate(row.decomposition),
        regime=LegalRegime.model_validate(row.regime),
        layer1_status=Layer1Status(row.layer1_status),
        model_confidence=row.model_confidence,
        self_consistency_votes=row.self_consistency_votes,
        created_at=row.created_at,
        evidence_spans=[
            EvidenceSpan(
                span_id=sp.span_id,
                element_id=sp.element_id,
                doc_id=sp.doc_id,
                char_start=sp.char_start,
                char_end=sp.char_end,
                role=SpanRole(sp.role),
            )
            for sp in spans
        ],
    )


def _to_coverage(row: CoverageRow) -> CoverageRecord:
    return CoverageRecord(
        jurisdiction=row.jurisdiction,
        indicator_id=row.indicator_id,
        state=CoverageState(row.state),
        measured_recall=row.measured_recall,
        reason=row.reason,
        verified_claim_ids=list(row.verified_claim_ids or []),
    )


def _to_layer2(row: ClaimRow) -> Layer2Recommendation | None:
    if row.layer2_recommendation is None:
        return None
    return Layer2Recommendation.model_validate(row.layer2_recommendation)
