from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.repositories.knowledge_base_repository import KnowledgeBaseRepository
from app.schemas.knowledge_base import (
    KnowledgeBaseCreateRequest,
    KnowledgeBaseResponse,
    KnowledgeBaseTenantTreeResponse,
    KnowledgeBaseTreeDocumentResponse,
    KnowledgeBaseTreeResponse,
)


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

    async def list_tree(
        self, *, keyword: str | None = None
    ) -> list[KnowledgeBaseTenantTreeResponse]:
        rows = await self.repository.list_tree(keyword=keyword)
        tenants: dict[str, KnowledgeBaseTenantTreeResponse] = {}
        knowledge_bases: dict[tuple[str, str], KnowledgeBaseTreeResponse] = {}

        for knowledge_base, document, chunk_count in rows:
            tenant = tenants.setdefault(
                knowledge_base.tenant_id,
                KnowledgeBaseTenantTreeResponse(
                    tenant_id=knowledge_base.tenant_id,
                    knowledge_bases=[],
                ),
            )
            knowledge_base_key = (knowledge_base.tenant_id, knowledge_base.id)
            tree_knowledge_base = knowledge_bases.get(knowledge_base_key)
            if tree_knowledge_base is None:
                tree_knowledge_base = KnowledgeBaseTreeResponse(
                    kb_id=knowledge_base.id,
                    name=knowledge_base.name,
                    description=knowledge_base.description,
                    created_at=knowledge_base.created_at,
                    documents=[],
                )
                knowledge_bases[knowledge_base_key] = tree_knowledge_base
                tenant.knowledge_bases.append(tree_knowledge_base)

            if document is not None:
                tree_knowledge_base.documents.append(
                    KnowledgeBaseTreeDocumentResponse(
                        document_id=document.id,
                        title=document.title,
                        chunk_count=chunk_count,
                        created_at=document.created_at,
                    )
                )

        return list(tenants.values())
