from datetime import datetime
from typing import TYPE_CHECKING, Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, DateTime, ForeignKey, Index, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.config import settings
from app.db.base import Base
from app.utils.id_generator import generate_id

if TYPE_CHECKING:
    from app.models.document import Document


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        Index("ix_chunks_tenant_kb", "tenant_id", "kb_id"),
        Index("ix_chunks_document_id", "document_id"),
        Index("ix_chunks_tenant_kb_version", "tenant_id", "kb_id", "index_version"),
        Index("ix_chunks_document_version_order", "document_id", "index_version", "order_index"),
        Index("ix_chunks_document_version_section", "document_id", "index_version", "section_id"),
        UniqueConstraint(
            "document_id",
            "index_version",
            "order_index",
            name="uq_chunks_document_version_order",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_id)
    tenant_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    kb_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False
    )
    document_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    index_version: Mapped[str] = mapped_column(
        String(32), nullable=False, default="v1", server_default="v1"
    )
    retrieval_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    section_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    parent_section_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    order_index: Mapped[int | None] = mapped_column(nullable=True)
    chunk_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSON, nullable=False, default=dict
    )
    embedding: Mapped[list[float]] = mapped_column(
        Vector(settings.embedding_dimensions), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    document: Mapped["Document"] = relationship(back_populates="chunks")
