"""add retrieval robustness metadata

Revision ID: 0009_retrieval_robust
Revises: 0008_multi_kb_retrieval
Create Date: 2026-09-20
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0009_retrieval_robust"
down_revision: str | None = "0008_multi_kb_retrieval"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "retrieval_logs",
        sa.Column("retrieval_metadata", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("retrieval_logs", "retrieval_metadata")
