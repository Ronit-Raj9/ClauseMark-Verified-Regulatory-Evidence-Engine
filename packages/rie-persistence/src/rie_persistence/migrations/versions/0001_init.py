"""Initial schema for the §10 RIE data model."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "documents",
        sa.Column("doc_id", sa.String(128), primary_key=True),
        sa.Column("jurisdiction", sa.String(64), nullable=False, index=True),
        sa.Column("title", sa.Text, nullable=False),
        sa.Column("document_type", sa.String(32), nullable=False),
        sa.Column("effective_date", sa.DateTime(timezone=True)),
        sa.Column("authority_tier", sa.String(32), nullable=False),
        sa.Column("source_url", sa.Text),
        sa.Column("sha256", sa.String(64), nullable=False, unique=True),
        sa.Column("retrieved_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("language", sa.String(16), nullable=False, server_default="en"),
    )
    op.create_table(
        "elements",
        sa.Column("element_id", sa.String(256), primary_key=True),
        sa.Column("doc_id", sa.String(128), sa.ForeignKey("documents.doc_id"), index=True),
        sa.Column("parent_id", sa.String(256)),
        sa.Column("element_type", sa.String(32), nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("page", sa.Integer, nullable=False),
        sa.Column("bbox", postgresql.JSONB),
        sa.Column("char_start", sa.Integer, nullable=False),
        sa.Column("char_end", sa.Integer, nullable=False),
        sa.Column("extraction_confidence", sa.Float, nullable=False),
        sa.Column("ocr_engine", sa.String(32), server_default="none"),
        sa.Column("corrected", sa.Boolean, server_default=sa.false()),
        sa.Column("legal_numbering", sa.String(64)),
    )
    op.create_index("ix_elements_doc_offset", "elements", ["doc_id", "char_start"])
    op.create_table(
        "structure_edges",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("from_element", sa.String(256), nullable=False, index=True),
        sa.Column("to_element", sa.String(256), nullable=False, index=True),
        sa.Column("edge_type", sa.String(32), nullable=False),
        sa.Column("raw_reference", sa.Text),
        sa.UniqueConstraint("from_element", "to_element", "edge_type"),
    )
    op.create_table(
        "claims",
        sa.Column("claim_id", sa.String(128), primary_key=True),
        sa.Column("indicator_id", sa.String(16), nullable=False, index=True),
        sa.Column("pillar_id", sa.String(8), nullable=False, index=True),
        sa.Column("clause_id", sa.String(256), nullable=False, index=True),
        sa.Column("jurisdiction", sa.String(64), nullable=False, index=True),
        sa.Column("clause_pattern", sa.String(32), nullable=False),
        sa.Column("decomposition", postgresql.JSONB, nullable=False),
        sa.Column("regime", postgresql.JSONB, nullable=False),
        sa.Column("layer1_status", sa.String(32), nullable=False, index=True),
        sa.Column("model_confidence", sa.Float),
        sa.Column("self_consistency_votes", postgresql.JSONB, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "evidence_spans",
        sa.Column("span_id", sa.String(256), primary_key=True),
        sa.Column("claim_id", sa.String(128), sa.ForeignKey("claims.claim_id"), index=True),
        sa.Column("element_id", sa.String(256), index=True),
        sa.Column("doc_id", sa.String(128), index=True),
        sa.Column("char_start", sa.Integer, nullable=False),
        sa.Column("char_end", sa.Integer, nullable=False),
        sa.Column("role", sa.String(32), nullable=False),
    )
    op.create_table(
        "verifications",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("claim_id", sa.String(128), sa.ForeignKey("claims.claim_id"), index=True),
        sa.Column("gate", sa.String(32), nullable=False),
        sa.Column("passed", sa.Boolean, nullable=False),
        sa.Column("detail", sa.Text, server_default=""),
        sa.Column("score", sa.Float),
        sa.Column("ran_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "coverage",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("jurisdiction", sa.String(64), nullable=False, index=True),
        sa.Column("indicator_id", sa.String(16), nullable=False, index=True),
        sa.Column("state", sa.String(64), nullable=False),
        sa.Column("measured_recall", sa.Float),
        sa.Column("reason", sa.Text),
        sa.Column("verified_claim_ids", sa.JSON, nullable=False),
        sa.UniqueConstraint("jurisdiction", "indicator_id"),
    )
    op.create_table(
        "reviews",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("claim_id", sa.String(128), sa.ForeignKey("claims.claim_id"), index=True),
        sa.Column("reviewer", sa.String(128), nullable=False),
        sa.Column("decision", sa.String(32), nullable=False),
        sa.Column("corrected_score", sa.String(8)),
        sa.Column("note", sa.Text, server_default=""),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "authority_overrides",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("jurisdiction", sa.String(64), nullable=False, index=True),
        sa.Column("source_pattern", sa.Text, nullable=False),
        sa.Column("authority_tier", sa.String(32), nullable=False),
        sa.Column("rationale", sa.Text, nullable=False),
    )


def downgrade() -> None:
    for t in [
        "authority_overrides",
        "reviews",
        "coverage",
        "verifications",
        "evidence_spans",
        "claims",
        "structure_edges",
        "elements",
        "documents",
    ]:
        op.drop_table(t)
