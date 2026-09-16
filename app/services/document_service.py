from sqlalchemy.ext.asyncio import AsyncSession

from app.core.error_codes import ErrorCode
from app.core.exceptions import (
    DocumentIndexingError,
    DocumentNotFoundError,
    raise_app_error,
)
from app.core.logging import get_logger
from app.models.document import DocumentStatus
from app.repositories.document_repository import DocumentRepository
from app.repositories.knowledge_base_repository import KnowledgeBaseRepository
from app.schemas.document import (
    DocumentDeleteResponse,
    DocumentDetailResponse,
    DocumentUploadRequest,
    DocumentUploadResponse,
)
from app.services.indexing_service import IndexingService
from app.tasks.indexing import index_document_task
from app.tenant.plan_resolver import PlanResolver


class DocumentService:
    def __init__(
        self,
        session: AsyncSession,
        document_repository: DocumentRepository,
        knowledge_base_repository: KnowledgeBaseRepository,
        indexing_service: IndexingService,
        plan_resolver: PlanResolver | None = None,
        quota_service: object | None = None,
    ) -> None:
        self.session = session
        self.document_repository = document_repository
        self.knowledge_base_repository = knowledge_base_repository
        self.indexing_service = indexing_service
        self.plan_resolver = plan_resolver or PlanResolver()
        self.quota_service = quota_service
        self.logger = get_logger(__name__)

    async def upload(
        self, request: DocumentUploadRequest, *, tenant_id: str
    ) -> DocumentUploadResponse:
        self.logger.info(
            "BUSINESS_EVENT | event=document_upload_started | kb_id=%s | tenant_id=%s | title=%s",
            request.kb_id,
            tenant_id,
            request.title,
        )
        if self.quota_service is not None:
            plan = await self.plan_resolver.resolve_for_tenant_id(tenant_id)
            await self.quota_service.check_upload_document(
                tenant_id=tenant_id,
                kb_id=request.kb_id,
                plan=plan,
            )
        document = await self.indexing_service.create_document_record(
            tenant_id,
            request,
        )
        try:
            index_document_task.delay(document.id)
        except Exception as exc:
            await self.indexing_service.mark_document_failed(document.id, str(exc))
            raise DocumentIndexingError(document.id, str(exc)) from exc

        return DocumentUploadResponse(
            document_id=document.id,
            kb_id=document.kb_id,
            status=int(DocumentStatus.PROCESSING),
            chunk_count=0,
        )

    async def get(self, document_id: str, *, tenant_id: str) -> DocumentDetailResponse:
        document = await self.document_repository.get_by_id_and_tenant(
            document_id=document_id,
            tenant_id=tenant_id,
        )
        if document is None:
            raise DocumentNotFoundError(document_id)
        return DocumentDetailResponse(
            document_id=document.id,
            kb_id=document.kb_id,
            title=document.title,
            status=int(document.status),
            error_message=document.error_message,
            chunk_count=await self.document_repository.count_chunks(document_id=document.id),
            created_at=document.created_at,
            updated_at=document.updated_at,
        )

    async def delete(self, document_id: str, *, tenant_id: str) -> DocumentDeleteResponse:
        document = await self.document_repository.get_by_id_and_tenant(
            document_id=document_id,
            tenant_id=tenant_id,
        )
        if document is None:
            raise DocumentNotFoundError(document_id)
        self._ensure_not_processing(document.status, operation="delete")

        try:
            await self.indexing_service.purge_document_chunks(document.id)
            await self.document_repository.delete_by_id(
                document_id=document.id,
                tenant_id=tenant_id,
            )
            await self.session.commit()
        except Exception:
            await self.session.rollback()
            raise
        self.logger.info(
            "BUSINESS_EVENT | event=document_deleted | document_id=%s | tenant_id=%s",
            document.id,
            tenant_id,
        )
        return DocumentDeleteResponse(document_id=document.id)

    async def reindex(
        self,
        document_id: str,
        *,
        tenant_id: str,
    ) -> DocumentUploadResponse:
        document = await self.document_repository.get_by_id_and_tenant(
            document_id=document_id,
            tenant_id=tenant_id,
        )
        if document is None:
            raise DocumentNotFoundError(document_id)
        if document.status != int(DocumentStatus.FAILED):
            raise_app_error(
                ErrorCode.PARAM_ERROR,
                "only failed documents can be reindexed",
                context={"document_id": document.id, "status": int(document.status)},
            )
        if self.quota_service is not None:
            plan = await self.plan_resolver.resolve_for_tenant_id(tenant_id)
            await self.quota_service.check_reindex_document(
                tenant_id=tenant_id,
                plan=plan,
            )

        try:
            await self.indexing_service.purge_document_chunks(document.id)
            document.status = int(DocumentStatus.PROCESSING)
            document.error_message = None
            await self.session.commit()
        except Exception:
            await self.session.rollback()
            raise
        try:
            index_document_task.delay(document.id)
        except Exception as exc:
            await self.indexing_service.mark_document_failed(document.id, str(exc))
            raise DocumentIndexingError(document.id, str(exc)) from exc

        return DocumentUploadResponse(
            document_id=document.id,
            kb_id=document.kb_id,
            status=int(DocumentStatus.PROCESSING),
            chunk_count=0,
        )

    @staticmethod
    def _ensure_not_processing(status: int, *, operation: str) -> None:
        if status == int(DocumentStatus.PROCESSING):
            raise_app_error(
                ErrorCode.PARAM_ERROR,
                f"cannot {operation} a processing document",
            )
