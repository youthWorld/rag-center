from typing import Any

from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chunk import Chunk
from app.models.document import Document, DocumentStatus
from app.models.knowledge_base import KnowledgeBase
from app.utils.id_generator import generate_id


class KnowledgeBaseRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self,
        *,
        tenant_id: str,
        name: str,
        description: str | None,
        settings: dict[str, Any] | None = None,
    ) -> KnowledgeBase:
        knowledge_base = KnowledgeBase(
            id=generate_id(),
            tenant_id=tenant_id,
            name=name,
            description=description,
            settings=dict(settings or {}),
        )
        self.session.add(knowledge_base)
        await self.session.flush()
        return knowledge_base

    async def get_by_id(self, *, kb_id: str, tenant_id: str) -> KnowledgeBase | None:
        statement = select(KnowledgeBase).where(
            KnowledgeBase.id == kb_id,
            KnowledgeBase.tenant_id == tenant_id,
        )
        result = await self.session.execute(statement)
        return result.scalar_one_or_none()

    async def get_settings(
        self,
        *,
        kb_id: str,
        tenant_id: str | None = None,
    ) -> dict[str, Any] | None:
        statement = select(KnowledgeBase).where(KnowledgeBase.id == kb_id)
        if tenant_id is not None:
            statement = statement.where(KnowledgeBase.tenant_id == tenant_id)
        result = await self.session.execute(statement)
        knowledge_base = result.scalar_one_or_none()
        if knowledge_base is None:
            return None
        return dict(knowledge_base.settings or {})

    async def update_settings(
        self,
        *,
        kb_id: str,
        settings: dict[str, Any],
        tenant_id: str | None = None,
    ) -> KnowledgeBase | None:
        statement = select(KnowledgeBase).where(KnowledgeBase.id == kb_id)
        if tenant_id is not None:
            statement = statement.where(KnowledgeBase.tenant_id == tenant_id)
        result = await self.session.execute(statement)
        knowledge_base = result.scalar_one_or_none()
        if knowledge_base is None:
            return None
        knowledge_base.settings = dict(settings)
        await self.session.flush()
        return knowledge_base

    async def set_settings(
        self,
        *,
        kb_id: str,
        settings: dict[str, Any],
        tenant_id: str | None = None,
    ) -> KnowledgeBase | None:
        return await self.update_settings(
            kb_id=kb_id,
            settings=settings,
            tenant_id=tenant_id,
        )

    async def list_tree(
        self, *, tenant_id: str, keyword: str | None = None
    ) -> list[tuple[KnowledgeBase, Document | None, int]]:
        statement = (
            select(KnowledgeBase, Document, func.count(Chunk.id).label("chunk_count"))
            .outerjoin(
                Document,
                and_(
                    Document.kb_id == KnowledgeBase.id,
                    Document.tenant_id == KnowledgeBase.tenant_id,
                    Document.status == int(DocumentStatus.SUCCESS),
                ),
            )
            .outerjoin(Chunk, Chunk.document_id == Document.id)
            .group_by(KnowledgeBase.id, Document.id)
            .order_by(
                KnowledgeBase.tenant_id.asc(),
                KnowledgeBase.created_at.desc(),
                KnowledgeBase.id.asc(),
                Document.created_at.desc(),
                Document.id.asc(),
            )
        )
        statement = statement.where(KnowledgeBase.tenant_id == tenant_id)
        normalized_keyword = keyword.strip() if keyword else ""
        if normalized_keyword:
            statement = statement.where(KnowledgeBase.name.ilike(f"%{normalized_keyword}%"))

        result = await self.session.execute(statement)
        return [
            (knowledge_base, document, int(chunk_count))
            for knowledge_base, document, chunk_count in result.all()
        ]
