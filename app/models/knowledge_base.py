from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import JSON, DateTime, String, Text, func, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.utils.id_generator import generate_id

if TYPE_CHECKING:
    from app.models.document import Document
    from app.models.index_version import KnowledgeBaseIndexVersion


class KnowledgeBase(Base):
    __tablename__ = "knowledge_bases"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_id)
    tenant_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    settings: Mapped[dict[str, Any]] = mapped_column(
        JSONB().with_variant(JSON(), "sqlite"),
        nullable=False,
        default=dict,
        server_default=text("'{}'"),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )
    active_index_version: Mapped[str] = mapped_column(
        String(32), nullable=False, default="v1", server_default=text("'v1'")
    )

    documents: Mapped[list["Document"]] = relationship(back_populates="knowledge_base")
    index_versions: Mapped[list["KnowledgeBaseIndexVersion"]] = relationship(
        back_populates="knowledge_base", cascade="all, delete-orphan"
    )
