from __future__ import annotations

from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, settings
from app.core.error_codes import ErrorCode
from app.core.exceptions import (
    DocumentIndexingError,
    DocumentNotFoundError,
    raise_app_error,
)
from app.core.logging import get_logger
from app.models.document import DocumentStatus
from app.providers.parsers.registry import is_supported_document, source_type_for_filename
from app.repositories.document_repository import DocumentRepository
from app.repositories.knowledge_base_repository import KnowledgeBaseRepository
from app.schemas.document import (
    DocumentDeleteResponse,
    DocumentDetailResponse,
    DocumentUploadRequest,
    DocumentUploadResponse,
)
from app.services.document_storage import DocumentStorage
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
        app_settings: Settings | None = None,
    ) -> None:
        self.session = session
        self.document_repository = document_repository
        self.knowledge_base_repository = knowledge_base_repository
        self.indexing_service = indexing_service
        self.plan_resolver = plan_resolver or PlanResolver()
        self.quota_service = quota_service
        self.settings = app_settings or settings
        self.document_storage = DocumentStorage(self.settings.document_storage_path)
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
        document = await self.indexing_service.create_document_record(tenant_id, request)
        await self._enqueue_document(document.id)
        return self._upload_response(document.id, document.kb_id)

    async def upload_file(
        self,
        *,
        tenant_id: str,
        kb_id: str,
        title: str | None,
        filename: str,
        mime_type: str | None,
        file_bytes: bytes,
    ) -> DocumentUploadResponse:
        try:
            normalized_filename = self.document_storage.normalize_filename(filename)
            self.document_storage.validate_component(tenant_id, label="tenant_id")
        except ValueError as exc:
            raise_app_error(ErrorCode.PARAM_ERROR, str(exc))
        if not is_supported_document(normalized_filename, mime_type):
            raise_app_error(
                ErrorCode.PARAM_ERROR,
                "unsupported document format: "
                f"{Path(normalized_filename).suffix or normalized_filename}",
                context={"filename": normalized_filename},
            )

        max_size = self.settings.document_max_size_mb * 1024 * 1024
        if len(file_bytes) > max_size:
            raise_app_error(
                ErrorCode.PARAM_ERROR,
                f"document exceeds the {self.settings.document_max_size_mb} MB size limit",
                context={"filename": normalized_filename, "size": len(file_bytes)},
            )

        normalized_title = (title or Path(normalized_filename).stem).strip()
        if not normalized_title:
            raise_app_error(ErrorCode.PARAM_ERROR, "document title must not be blank")
        if len(normalized_title) > 255:
            raise_app_error(ErrorCode.PARAM_ERROR, "document title is too long")
        source_type = source_type_for_filename(normalized_filename)
        if source_type is None:
            raise_app_error(ErrorCode.PARAM_ERROR, "unsupported document format")

        await self._check_upload_quota(tenant_id=tenant_id, kb_id=kb_id)
        document = await self.indexing_service.create_file_document_record(
            tenant_id=tenant_id,
            kb_id=kb_id,
            title=normalized_title,
            source_type=source_type,
            source_filename=normalized_filename,
        )
        source_path = self.document_storage.path_for(
            tenant_id=tenant_id,
            document_id=document.id,
            filename=normalized_filename,
        )
        source_path_committed = False
        try:
            await self.document_storage.write_bytes(source_path, file_bytes)
            document.source_file_path = str(source_path)
            document.source_filename = normalized_filename
            await self.session.commit()
            source_path_committed = True
            await self.session.refresh(document)
            await self._enqueue_document(document.id)
        except DocumentIndexingError:
            raise
        except Exception as exc:
            if not source_path_committed:
                await self.document_storage.remove(
                    tenant_id=tenant_id,
                    document_id=document.id,
                    source_file_path=str(source_path),
                )
            await self.indexing_service.mark_document_failed(document.id, str(exc))
            raise DocumentIndexingError(document.id, str(exc)) from exc

        return self._upload_response(document.id, document.kb_id)

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
            source_type=getattr(document, "source_type", "text"),
            source_filename=getattr(document, "source_filename", None),
        )

    async def delete(self, document_id: str, *, tenant_id: str) -> DocumentDeleteResponse:
        document = await self.document_repository.get_by_id_and_tenant(
            document_id=document_id,
            tenant_id=tenant_id,
        )
        if document is None:
            raise DocumentNotFoundError(document_id)
        self._ensure_not_processing(document.status, operation="delete")
        await self._ensure_no_rebuild(tenant_id=tenant_id, kb_id=document.kb_id)

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
        await self.document_storage.remove(
            tenant_id=tenant_id,
            document_id=document.id,
            source_file_path=getattr(document, "source_file_path", None),
        )
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
        reparse: bool = False,
    ) -> DocumentUploadResponse:
        document = await self.document_repository.get_by_id_and_tenant(
            document_id=document_id,
            tenant_id=tenant_id,
        )
        if document is None:
            raise DocumentNotFoundError(document_id)
        if document.status == int(DocumentStatus.PROCESSING):
            raise_app_error(
                ErrorCode.PARAM_ERROR,
                "cannot reindex a processing document",
                context={"document_id": document.id, "status": int(document.status)},
            )
        if document.status not in {
            int(DocumentStatus.SUCCESS),
            int(DocumentStatus.FAILED),
        }:
            raise_app_error(
                ErrorCode.PARAM_ERROR,
                "only successful or failed documents can be reindexed",
                context={"document_id": document.id, "status": int(document.status)},
            )
        await self._ensure_no_rebuild(tenant_id=tenant_id, kb_id=document.kb_id)
        if self.quota_service is not None:
            plan = await self.plan_resolver.resolve_for_tenant_id(tenant_id)
            await self.quota_service.check_reindex_document(
                tenant_id=tenant_id,
                plan=plan,
            )

        try:
            get_kb = getattr(self.knowledge_base_repository, "get_by_id", None)
            knowledge_base = (
                await get_kb(kb_id=document.kb_id, tenant_id=tenant_id)
                if callable(get_kb)
                else None
            )
            active_version = getattr(knowledge_base, "active_index_version", "v1") or "v1"
            if callable(get_kb):
                try:
                    await self.indexing_service.purge_document_chunks(
                        document.id, index_version=active_version
                    )
                except TypeError as exception:
                    if "index_version" not in str(exception):
                        raise
                    await self.indexing_service.purge_document_chunks(document.id)
            else:
                await self.indexing_service.purge_document_chunks(document.id)
            document.status = int(DocumentStatus.PROCESSING)
            document.error_message = None
            await self.session.commit()
        except Exception:
            await self.session.rollback()
            raise
        if reparse and getattr(document, "source_file_path", None):
            try:
                index_document_task.delay(document.id, reparse=True)
            except Exception as exc:
                await self.indexing_service.mark_document_failed(document.id, str(exc))
                raise DocumentIndexingError(document.id, str(exc)) from exc
        else:
            await self._enqueue_document(document.id)

        return self._upload_response(document.id, document.kb_id)

    async def _ensure_no_rebuild(self, *, tenant_id: str, kb_id: str) -> None:
        guard = getattr(self.indexing_service, "_ensure_not_building", None)
        if callable(guard):
            await guard(tenant_id=tenant_id, kb_id=kb_id)

    async def _check_upload_quota(self, *, tenant_id: str, kb_id: str) -> None:
        if self.quota_service is None:
            return
        plan = await self.plan_resolver.resolve_for_tenant_id(tenant_id)
        await self.quota_service.check_upload_document(
            tenant_id=tenant_id,
            kb_id=kb_id,
            plan=plan,
        )

    async def _enqueue_document(self, document_id: str) -> None:
        try:
            index_document_task.delay(document_id)
        except Exception as exc:
            await self.indexing_service.mark_document_failed(document_id, str(exc))
            raise DocumentIndexingError(document_id, str(exc)) from exc

    @staticmethod
    def _upload_response(document_id: str, kb_id: str) -> DocumentUploadResponse:
        return DocumentUploadResponse(
            document_id=document_id,
            kb_id=kb_id,
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
