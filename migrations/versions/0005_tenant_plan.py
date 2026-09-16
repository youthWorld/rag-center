"""add tenant plans and quota policies

Revision ID: 0005_tenant_plan
Revises: 0003_query_semantic_optimization
Create Date: 2026-09-16
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005_tenant_plan"
down_revision: str | None = "0003_query_semantic_optimization"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "tenants",
        sa.Column("plan", sa.String(length=32), nullable=False, server_default="free"),
    )
    op.execute("UPDATE tenants SET plan = 'standard'")
    op.execute("UPDATE tenants SET plan = 'pro' WHERE id = 'tenant_demo'")
    op.alter_column("tenants", "plan", server_default=None)


def downgrade() -> None:
    op.drop_column("tenants", "plan")
