"""add multi-knowledge-base retrieval log metadata

Revision ID: 0008_multi_kb_retrieval
Revises: 0007_document_source_files
Create Date: 2026-09-19
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0008_multi_kb_retrieval"
down_revision: str | None = "0007_document_source_files"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "retrieval_logs",
        sa.Column("kb_ids", postgresql.JSONB(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("retrieval_logs", "kb_ids")
