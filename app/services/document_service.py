from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import DocumentIndexingError, KnowledgeBaseNotFoundError
from app.core.logging import get_logger
from app.models.document import DocumentStatus
from app.repositories.document_repository import DocumentRepository
from app.repositories.knowledge_base_repository import KnowledgeBaseRepository
from app.schemas.document import DocumentUploadRequest, DocumentUploadResponse
from app.services.indexing_service import IndexingService


class DocumentService:
    def __init__(
        self,
        session: AsyncSession,
        document_repository: DocumentRepository,
        knowledge_base_repository: KnowledgeBaseRepository,
        indexing_service: IndexingService,
    ) -> None:
        self.session = session
        self.document_repository = document_repository
        self.knowledge_base_repository = knowledge_base_repository
        self.indexing_service = indexing_service
        self.logger = get_logger(__name__)

    async def upload(self, request: DocumentUploadRequest) -> DocumentUploadResponse:
        self.logger.info(
            "BUSINESS_EVENT | event=document_upload_started | kb_id=%s | tenant_id=%s | title=%s",
            request.kb_id,
            request.tenant_id,
            request.title,
        )
        knowledge_base = await self.knowledge_base_repository.get_by_id(
            kb_id=request.kb_id,
            tenant_id=request.tenant_id,
        )
        if knowledge_base is None:
            raise KnowledgeBaseNotFoundError()

        document = await self.document_repository.create(
            tenant_id=request.tenant_id,
            kb_id=knowledge_base.id,
            title=request.title,
            content=request.content,
        )

        try:
            async with self.session.begin_nested():
                chunk_count = await self.indexing_service.index_document(document)
            document.status = int(DocumentStatus.SUCCESS)
            document.error_message = None
            await self.session.commit()
        except Exception as exc:
            self.logger.exception(
                "BUSINESS_ERROR | event=document_indexing_failed | document_id=%s",
                document.id,
            )
            document.status = int(DocumentStatus.FAILED)
            document.error_message = str(exc)[:2000]
            await self.session.commit()
            raise DocumentIndexingError(document.id, str(exc)) from exc

        self.logger.info(
            "BUSINESS_EVENT | event=document_indexed | document_id=%s | chunk_count=%s",
            document.id,
            chunk_count,
        )
        return DocumentUploadResponse(
            document_id=document.id,
            kb_id=document.kb_id,
            status=int(document.status),
            chunk_count=chunk_count,
        )
