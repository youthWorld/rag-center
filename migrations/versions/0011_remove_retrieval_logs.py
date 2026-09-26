"""remove the local retrieval log table

Revision ID: 0011_remove_retrieval_logs
Revises: 0010_context_graph_indexing
Create Date: 2026-09-26
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0011_remove_retrieval_logs"
down_revision: str | None = "0010_context_graph_indexing"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_index("ix_retrieval_logs_user_id", table_name="retrieval_logs")
    op.drop_index("ix_retrieval_logs_kb_id", table_name="retrieval_logs")
    op.drop_index("ix_retrieval_logs_tenant_id", table_name="retrieval_logs")
    op.drop_table("retrieval_logs")


def downgrade() -> None:
    jsonb_or_json = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")
    op.create_table(
        "retrieval_logs",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("tenant_id", sa.String(length=128), nullable=False),
        sa.Column("kb_id", sa.String(length=36), nullable=False),
        sa.Column("kb_ids", jsonb_or_json, nullable=True),
        sa.Column("user_id", sa.String(length=128), nullable=False),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("trace_id", sa.String(length=64), nullable=True),
        sa.Column("profile", sa.String(length=32), nullable=True),
        sa.Column("search_query", sa.Text(), nullable=True),
        sa.Column("effective_query", sa.Text(), nullable=True),
        sa.Column("retrieved_chunks", sa.JSON(), nullable=False),
        sa.Column("retrieval_metadata", sa.JSON(), nullable=True),
        sa.Column("top_k", sa.Integer(), nullable=False),
        sa.Column("vector_store", sa.String(length=64), nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(["kb_id"], ["knowledge_bases.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_retrieval_logs_tenant_id", "retrieval_logs", ["tenant_id"])
    op.create_index("ix_retrieval_logs_kb_id", "retrieval_logs", ["kb_id"])
    op.create_index("ix_retrieval_logs_user_id", "retrieval_logs", ["user_id"])
