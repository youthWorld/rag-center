"""add document source file metadata

Revision ID: 0007_document_source_files
Revises: 0006_retrieval_log_observability
Create Date: 2026-09-19
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0007_document_source_files"
down_revision: str | None = "0006_retrieval_log_observability"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("source_file_path", sa.Text(), nullable=True))
    op.add_column(
        "documents",
        sa.Column("source_filename", sa.String(length=255), nullable=True),
    )
    op.alter_column(
        "documents",
        "content",
        existing_type=sa.Text(),
        nullable=True,
    )


def downgrade() -> None:
    op.alter_column(
        "documents",
        "content",
        existing_type=sa.Text(),
        nullable=False,
    )
    op.drop_column("documents", "source_filename")
    op.drop_column("documents", "source_file_path")
