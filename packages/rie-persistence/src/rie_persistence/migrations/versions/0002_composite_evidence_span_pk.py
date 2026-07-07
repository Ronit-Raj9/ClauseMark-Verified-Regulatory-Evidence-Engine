"""Composite pkey on evidence_spans — span_id may appear in multiple claims."""

from __future__ import annotations

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint("evidence_spans_pkey", "evidence_spans", type_="primary")
    op.create_primary_key("evidence_spans_pkey", "evidence_spans", ["span_id", "claim_id"])


def downgrade() -> None:
    op.drop_constraint("evidence_spans_pkey", "evidence_spans", type_="primary")
    op.create_primary_key("evidence_spans_pkey", "evidence_spans", ["span_id"])
