from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.repositories.knowledge_base_repository import KnowledgeBaseRepository
from app.schemas.knowledge_base import KnowledgeBaseCreateRequest, KnowledgeBaseResponse


class KnowledgeBaseService:
    def __init__(self, session: AsyncSession, repository: KnowledgeBaseRepository) -> None:
        self.session = session
        self.repository = repository
        self.logger = get_logger(__name__)

    async def create(self, request: KnowledgeBaseCreateRequest) -> KnowledgeBaseResponse:
        knowledge_base = await self.repository.create(
            tenant_id=request.tenant_id,
            name=request.name,
            description=request.description,
        )
        await self.session.commit()
        await self.session.refresh(knowledge_base)
        self.logger.info(
            "BUSINESS_EVENT | event=knowledge_base_created | kb_id=%s | tenant_id=%s",
            knowledge_base.id,
            knowledge_base.tenant_id,
        )
        return KnowledgeBaseResponse(
            kb_id=knowledge_base.id,
            name=knowledge_base.name,
            tenant_id=knowledge_base.tenant_id,
            created_at=knowledge_base.created_at,
        )
