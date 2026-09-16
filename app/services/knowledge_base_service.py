from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.error_codes import ErrorCode
from app.core.exceptions import KnowledgeBaseNotFoundError, raise_app_error
from app.core.logging import get_logger
from app.models.document import DocumentStatus
from app.repositories.document_repository import DocumentRepository
from app.repositories.knowledge_base_repository import KnowledgeBaseRepository
from app.schemas.knowledge_base import (
    KnowledgeBaseCreateRequest,
    KnowledgeBaseDeleteResponse,
    KnowledgeBaseDetailResponse,
    KnowledgeBaseResponse,
    KnowledgeBaseTenantTreeResponse,
    KnowledgeBaseTreeDocumentResponse,
    KnowledgeBaseTreeResponse,
    KnowledgeBaseUpdateRequest,
)
from app.services.indexing_service import IndexingService
from app.utils.knowledge_base_settings import validate_knowledge_base_settings


class KnowledgeBaseService:
    def __init__(
        self,
        session: AsyncSession,
        repository: KnowledgeBaseRepository,
        document_repository: DocumentRepository | None = None,
        indexing_service: IndexingService | None = None,
    ) -> None:
        self.session = session
        self.repository = repository
        self.document_repository = document_repository
        self.indexing_service = indexing_service
        self.logger = get_logger(__name__)

    async def create(
        self, request: KnowledgeBaseCreateRequest, *, tenant_id: str
    ) -> KnowledgeBaseResponse:
        knowledge_base = await self.repository.create(
            tenant_id=tenant_id,
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

    async def get(self, kb_id: str, *, tenant_id: str) -> KnowledgeBaseDetailResponse:
        knowledge_base = await self.repository.get_by_id_and_tenant(
            kb_id=kb_id,
            tenant_id=tenant_id,
        )
        if knowledge_base is None:
            raise KnowledgeBaseNotFoundError(kb_id)
        return await self._to_detail(knowledge_base, tenant_id=tenant_id)

    async def update(
        self,
        kb_id: str,
        request: KnowledgeBaseUpdateRequest,
        *,
        tenant_id: str,
    ) -> KnowledgeBaseDetailResponse:
        knowledge_base = await self.repository.get_by_id_and_tenant(
            kb_id=kb_id,
            tenant_id=tenant_id,
        )
        if knowledge_base is None:
            raise KnowledgeBaseNotFoundError(kb_id)
        if not request.model_fields_set:
            raise_app_error(ErrorCode.PARAM_ERROR, "at least one field is required")

        if "name" in request.model_fields_set:
            if request.name is None or not request.name.strip():
                raise_app_error(ErrorCode.PARAM_ERROR, "name must not be blank")
            knowledge_base.name = request.name.strip()
        if "description" in request.model_fields_set:
            knowledge_base.description = (
                request.description.strip() if request.description is not None else None
            )
        if "settings" in request.model_fields_set:
            if not isinstance(request.settings, dict):
                raise_app_error(ErrorCode.PARAM_ERROR, "settings must be an object")
            try:
                knowledge_base.settings = validate_knowledge_base_settings(request.settings)
            except ValueError as exc:
                raise_app_error(
                    ErrorCode.PARAM_ERROR,
                    str(exc),
                    context={"kb_id": kb_id, "field": "settings"},
                )

        await self.session.commit()
        await self.session.refresh(knowledge_base)
        self.logger.info(
            "BUSINESS_EVENT | event=knowledge_base_updated | kb_id=%s | tenant_id=%s",
            kb_id,
            tenant_id,
        )
        return await self._to_detail(knowledge_base, tenant_id=tenant_id)

    async def delete(self, kb_id: str, *, tenant_id: str) -> KnowledgeBaseDeleteResponse:
        knowledge_base = await self.repository.get_by_id_and_tenant(
            kb_id=kb_id,
            tenant_id=tenant_id,
        )
        if knowledge_base is None:
            raise KnowledgeBaseNotFoundError(kb_id)
        self._require_maintenance_dependencies()

        documents = await self.document_repository.list_by_kb_id(
            kb_id=kb_id,
            tenant_id=tenant_id,
        )
        if any(document.status == int(DocumentStatus.PROCESSING) for document in documents):
            raise_app_error(
                ErrorCode.PARAM_ERROR,
                "cannot delete a knowledge base with processing documents",
                context={"kb_id": kb_id},
            )

        try:
            for document in documents:
                await self.indexing_service.purge_document_chunks(document.id)
            await self.document_repository.delete_by_kb_id(kb_id=kb_id, tenant_id=tenant_id)
            await self.repository.delete_by_id(kb_id=kb_id, tenant_id=tenant_id)
            await self.session.commit()
        except Exception:
            await self.session.rollback()
            raise
        self.logger.info(
            "BUSINESS_EVENT | event=knowledge_base_deleted | kb_id=%s | tenant_id=%s",
            kb_id,
            tenant_id,
        )
        return KnowledgeBaseDeleteResponse(kb_id=kb_id)

    async def list_tree(
        self, *, tenant_id: str, keyword: str | None = None
    ) -> list[KnowledgeBaseTenantTreeResponse]:
        rows = await self.repository.list_tree(tenant_id=tenant_id, keyword=keyword)
        tenants: dict[str, KnowledgeBaseTenantTreeResponse] = {
            tenant_id: KnowledgeBaseTenantTreeResponse(
                tenant_id=tenant_id,
                knowledge_bases=[],
            )
        }
        knowledge_bases: dict[tuple[str, str], KnowledgeBaseTreeResponse] = {}

        for knowledge_base, document, chunk_count in rows:
            if knowledge_base.tenant_id != tenant_id:
                continue
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
                status = DocumentStatus(
                    int(getattr(document, "status", int(DocumentStatus.SUCCESS)))
                ).name
                tree_knowledge_base.documents.append(
                    KnowledgeBaseTreeDocumentResponse(
                        document_id=document.id,
                        title=document.title,
                        status=status,
                        error_message=getattr(document, "error_message", None),
                        chunk_count=chunk_count,
                        created_at=document.created_at,
                    )
                )

        return [tenants[tenant_id]]

    async def _to_detail(
        self,
        knowledge_base: Any,
        *,
        tenant_id: str,
    ) -> KnowledgeBaseDetailResponse:
        return KnowledgeBaseDetailResponse(
            kb_id=knowledge_base.id,
            name=knowledge_base.name,
            description=knowledge_base.description,
            settings=dict(knowledge_base.settings or {}),
            document_count=await self.repository.count_documents(
                kb_id=knowledge_base.id,
                tenant_id=tenant_id,
            ),
            created_at=knowledge_base.created_at,
            updated_at=knowledge_base.updated_at,
        )

    def _require_maintenance_dependencies(self) -> None:
        if self.document_repository is None or self.indexing_service is None:
            raise RuntimeError("knowledge base maintenance dependencies are not configured")
