from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.knowledge_base import KnowledgeBase
from app.utils.id_generator import generate_id


class KnowledgeBaseRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, *, tenant_id: str, name: str, description: str | None) -> KnowledgeBase:
        knowledge_base = KnowledgeBase(
            id=generate_id(),
            tenant_id=tenant_id,
            name=name,
            description=description,
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
