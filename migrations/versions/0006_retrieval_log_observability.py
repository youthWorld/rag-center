"""add retrieval observability fields

Revision ID: 0006_retrieval_log_observability
Revises: 0005_tenant_plan
Create Date: 2026-09-17
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006_retrieval_log_observability"
down_revision: str | None = "0005_tenant_plan"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("retrieval_logs", sa.Column("trace_id", sa.String(length=64), nullable=True))
    op.add_column("retrieval_logs", sa.Column("profile", sa.String(length=32), nullable=True))
    op.add_column("retrieval_logs", sa.Column("search_query", sa.Text(), nullable=True))
    op.add_column("retrieval_logs", sa.Column("effective_query", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("retrieval_logs", "effective_query")
    op.drop_column("retrieval_logs", "search_query")
    op.drop_column("retrieval_logs", "profile")
    op.drop_column("retrieval_logs", "trace_id")
