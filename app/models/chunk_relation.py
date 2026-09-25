from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.utils.id_generator import generate_id


class ChunkRelation(Base):
    __tablename__ = "chunk_relations"
    __table_args__ = (
        CheckConstraint(
            "source_chunk_id <> target_chunk_id", name="ck_chunk_relations_distinct_chunks"
        ),
        UniqueConstraint(
            "tenant_id",
            "kb_id",
            "index_version",
            "source_chunk_id",
            "target_chunk_id",
            "relation_type",
            name="uq_chunk_relations_edge",
        ),
        Index(
            "ix_chunk_relations_source",
            "tenant_id",
            "kb_id",
            "index_version",
            "source_chunk_id",
        ),
        Index(
            "ix_chunk_relations_target",
            "tenant_id",
            "kb_id",
            "index_version",
            "target_chunk_id",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_id)
    tenant_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    kb_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    index_version: Mapped[str] = mapped_column(String(32), nullable=False)
    source_chunk_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("chunks.id", ondelete="CASCADE"), nullable=False
    )
    target_chunk_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("chunks.id", ondelete="CASCADE"), nullable=False
    )
    relation_type: Mapped[str] = mapped_column(String(32), nullable=False, default="reference")
    relation_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSON, nullable=False, default=dict
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
