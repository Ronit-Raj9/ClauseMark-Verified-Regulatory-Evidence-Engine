"""SQLAlchemy ORM mapping for the §10 data model."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB as _PG_JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Use Postgres JSONB at runtime, generic JSON for sqlite smoke tests.
JSONB = _PG_JSONB().with_variant(JSON(), "sqlite")


class Base(DeclarativeBase):
    pass


class DocumentRow(Base):
    __tablename__ = "documents"
    doc_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    jurisdiction: Mapped[str] = mapped_column(String(64), index=True)
    title: Mapped[str] = mapped_column(Text)
    document_type: Mapped[str] = mapped_column(String(32))
    effective_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    authority_tier: Mapped[str] = mapped_column(String(32))
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    sha256: Mapped[str] = mapped_column(String(64), unique=True)
    retrieved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    language: Mapped[str] = mapped_column(String(16), default="en")


class ElementRow(Base):
    __tablename__ = "elements"
    element_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    doc_id: Mapped[str] = mapped_column(ForeignKey("documents.doc_id"), index=True)
    parent_id: Mapped[str | None] = mapped_column(String(256), nullable=True)
    element_type: Mapped[str] = mapped_column(String(32))
    text: Mapped[str] = mapped_column(Text)
    page: Mapped[int] = mapped_column(Integer)
    bbox: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    char_start: Mapped[int] = mapped_column(Integer)
    char_end: Mapped[int] = mapped_column(Integer)
    extraction_confidence: Mapped[float] = mapped_column(Float)
    ocr_engine: Mapped[str] = mapped_column(String(32), default="none")
    corrected: Mapped[bool] = mapped_column(Boolean, default=False)
    legal_numbering: Mapped[str | None] = mapped_column(String(64), nullable=True)


class StructureEdgeRow(Base):
    __tablename__ = "structure_edges"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    from_element: Mapped[str] = mapped_column(String(256), index=True)
    to_element: Mapped[str] = mapped_column(String(256), index=True)
    edge_type: Mapped[str] = mapped_column(String(32))
    raw_reference: Mapped[str | None] = mapped_column(Text, nullable=True)
    __table_args__ = (UniqueConstraint("from_element", "to_element", "edge_type"),)


class ClaimRow(Base):
    __tablename__ = "claims"
    claim_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    indicator_id: Mapped[str] = mapped_column(String(16), index=True)
    pillar_id: Mapped[str] = mapped_column(String(8), index=True)
    clause_id: Mapped[str] = mapped_column(String(256), index=True)
    jurisdiction: Mapped[str] = mapped_column(String(64), index=True)
    clause_pattern: Mapped[str] = mapped_column(String(32))
    decomposition: Mapped[dict[str, Any]] = mapped_column(JSONB)
    regime: Mapped[dict[str, Any]] = mapped_column(JSONB)
    layer1_status: Mapped[str] = mapped_column(String(32), index=True)
    model_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    self_consistency_votes: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    layer2_recommendation: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    spans = relationship("EvidenceSpanRow", back_populates="claim", cascade="all, delete-orphan")


class EvidenceSpanRow(Base):
    __tablename__ = "evidence_spans"
    # Composite PK — same span_id may legitimately appear in multiple claims as
    # a regime member or shared definition reference.
    span_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    claim_id: Mapped[str] = mapped_column(
        ForeignKey("claims.claim_id"), primary_key=True, index=True
    )
    element_id: Mapped[str] = mapped_column(String(256), index=True)
    doc_id: Mapped[str] = mapped_column(String(128), index=True)
    char_start: Mapped[int] = mapped_column(Integer)
    char_end: Mapped[int] = mapped_column(Integer)
    role: Mapped[str] = mapped_column(String(32))
    claim = relationship("ClaimRow", back_populates="spans")


class VerificationRow(Base):
    __tablename__ = "verifications"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    claim_id: Mapped[str] = mapped_column(ForeignKey("claims.claim_id"), index=True)
    gate: Mapped[str] = mapped_column(String(32))
    passed: Mapped[bool] = mapped_column(Boolean)
    detail: Mapped[str] = mapped_column(Text, default="")
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    ran_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class CoverageRow(Base):
    __tablename__ = "coverage"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    jurisdiction: Mapped[str] = mapped_column(String(64), index=True)
    indicator_id: Mapped[str] = mapped_column(String(16), index=True)
    state: Mapped[str] = mapped_column(String(64))
    measured_recall: Mapped[float | None] = mapped_column(Float, nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    verified_claim_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    __table_args__ = (UniqueConstraint("jurisdiction", "indicator_id"),)


class ReviewRow(Base):
    __tablename__ = "reviews"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    claim_id: Mapped[str] = mapped_column(ForeignKey("claims.claim_id"), index=True)
    reviewer: Mapped[str] = mapped_column(String(128))
    decision: Mapped[str] = mapped_column(String(32))
    corrected_score: Mapped[str | None] = mapped_column(String(8), nullable=True)
    note: Mapped[str] = mapped_column(Text, default="")
    decided_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class AuthorityOverrideRow(Base):
    __tablename__ = "authority_overrides"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    jurisdiction: Mapped[str] = mapped_column(String(64), index=True)
    source_pattern: Mapped[str] = mapped_column(Text)
    authority_tier: Mapped[str] = mapped_column(String(32))
    rationale: Mapped[str] = mapped_column(Text)
