"""add versioned contextual chunks and lightweight chunk relations

Revision ID: 0010_context_graph_indexing
Revises: 0009_retrieval_robust
Create Date: 2026-09-24
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0010_context_graph_indexing"
down_revision: str | None = "0009_retrieval_robust"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "knowledge_bases",
        sa.Column("active_index_version", sa.String(length=32), nullable=True),
    )
    op.execute(
        "UPDATE knowledge_bases SET active_index_version = 'v1' WHERE active_index_version IS NULL"
    )
    op.alter_column(
        "knowledge_bases",
        "active_index_version",
        nullable=False,
        server_default="v1",
    )

    op.create_table(
        "knowledge_base_index_versions",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("tenant_id", sa.String(length=128), nullable=False),
        sa.Column("kb_id", sa.String(length=36), nullable=False),
        sa.Column("version", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("document_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("chunk_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("activated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["kb_id"], ["knowledge_bases.id"], ondelete="CASCADE"),
        sa.CheckConstraint(
            "status IN ('building', 'ready', 'active', 'failed', 'archived')",
            name="ck_kb_index_versions_status",
        ),
        sa.UniqueConstraint(
            "tenant_id", "kb_id", "version", name="uq_kb_index_versions_tenant_kb_version"
        ),
    )
    op.create_index(
        "ix_knowledge_base_index_versions_tenant_id",
        "knowledge_base_index_versions",
        ["tenant_id"],
    )
    op.create_index(
        "uq_kb_index_versions_active",
        "knowledge_base_index_versions",
        ["tenant_id", "kb_id"],
        unique=True,
        postgresql_where=sa.text("status = 'active'"),
        sqlite_where=sa.text("status = 'active'"),
    )

    op.add_column("chunks", sa.Column("index_version", sa.String(length=32), nullable=True))
    op.add_column("chunks", sa.Column("retrieval_text", sa.Text(), nullable=True))
    op.add_column("chunks", sa.Column("section_id", sa.String(length=128), nullable=True))
    op.add_column("chunks", sa.Column("parent_section_id", sa.String(length=128), nullable=True))
    op.add_column("chunks", sa.Column("order_index", sa.Integer(), nullable=True))
    op.execute("UPDATE chunks SET index_version = 'v1' WHERE index_version IS NULL")
    op.alter_column("chunks", "index_version", nullable=False, server_default="v1")
    op.create_index(
        "ix_chunks_tenant_kb_version",
        "chunks",
        ["tenant_id", "kb_id", "index_version"],
    )
    op.create_index(
        "ix_chunks_document_version_order",
        "chunks",
        ["document_id", "index_version", "order_index"],
    )
    op.create_index(
        "ix_chunks_document_version_section",
        "chunks",
        ["document_id", "index_version", "section_id"],
    )
    op.create_unique_constraint(
        "uq_chunks_document_version_order",
        "chunks",
        ["document_id", "index_version", "order_index"],
    )

    op.create_table(
        "chunk_relations",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("tenant_id", sa.String(length=128), nullable=False),
        sa.Column("kb_id", sa.String(length=36), nullable=False),
        sa.Column("index_version", sa.String(length=32), nullable=False),
        sa.Column("source_chunk_id", sa.String(length=36), nullable=False),
        sa.Column("target_chunk_id", sa.String(length=36), nullable=False),
        sa.Column("relation_type", sa.String(length=32), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.ForeignKeyConstraint(["source_chunk_id"], ["chunks.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["target_chunk_id"], ["chunks.id"], ondelete="CASCADE"),
        sa.CheckConstraint(
            "source_chunk_id <> target_chunk_id", name="ck_chunk_relations_distinct_chunks"
        ),
        sa.UniqueConstraint(
            "tenant_id",
            "kb_id",
            "index_version",
            "source_chunk_id",
            "target_chunk_id",
            "relation_type",
            name="uq_chunk_relations_edge",
        ),
    )
    op.create_index(
        "ix_chunk_relations_tenant_id",
        "chunk_relations",
        ["tenant_id"],
    )
    op.create_index(
        "ix_chunk_relations_kb_id",
        "chunk_relations",
        ["kb_id"],
    )
    op.create_index(
        "ix_chunk_relations_source",
        "chunk_relations",
        ["tenant_id", "kb_id", "index_version", "source_chunk_id"],
    )
    op.create_index(
        "ix_chunk_relations_target",
        "chunk_relations",
        ["tenant_id", "kb_id", "index_version", "target_chunk_id"],
    )

    op.execute(
        "INSERT INTO knowledge_base_index_versions "
        "(id, tenant_id, kb_id, version, status, document_count, chunk_count, activated_at) "
        "SELECT kb.id, kb.tenant_id, kb.id, 'v1', 'active', "
        "(SELECT count(*) FROM documents d WHERE d.kb_id = kb.id), "
        "(SELECT count(*) FROM chunks c WHERE c.kb_id = kb.id AND c.index_version = 'v1'), "
        "CURRENT_TIMESTAMP FROM knowledge_bases kb"
    )


def downgrade() -> None:
    op.drop_index("ix_chunk_relations_target", table_name="chunk_relations")
    op.drop_index("ix_chunk_relations_source", table_name="chunk_relations")
    op.drop_index("ix_chunk_relations_kb_id", table_name="chunk_relations")
    op.drop_index("ix_chunk_relations_tenant_id", table_name="chunk_relations")
    op.drop_table("chunk_relations")

    op.drop_constraint("uq_chunks_document_version_order", "chunks", type_="unique")
    op.drop_index("ix_chunks_document_version_section", table_name="chunks")
    op.drop_index("ix_chunks_document_version_order", table_name="chunks")
    op.drop_index("ix_chunks_tenant_kb_version", table_name="chunks")
    for column in (
        "order_index",
        "parent_section_id",
        "section_id",
        "retrieval_text",
        "index_version",
    ):
        op.drop_column("chunks", column)

    op.drop_index("uq_kb_index_versions_active", table_name="knowledge_base_index_versions")
    op.drop_index(
        "ix_knowledge_base_index_versions_tenant_id",
        table_name="knowledge_base_index_versions",
    )
    op.drop_table("knowledge_base_index_versions")
    op.drop_column("knowledge_bases", "active_index_version")
