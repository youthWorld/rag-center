from __future__ import annotations

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.error_codes import ErrorCode
from app.core.exceptions import KnowledgeBaseNotFoundError, raise_app_error
from app.models.index_version import IndexVersionStatus
from app.repositories.document_repository import DocumentRepository
from app.repositories.index_version_repository import IndexVersionRepository
from app.repositories.knowledge_base_repository import KnowledgeBaseRepository
from app.schemas.index_version import (
    IndexVersionListResponse,
    IndexVersionRebuildResponse,
    IndexVersionResponse,
)
from app.services.indexing_service import IndexingService
from app.tasks.indexing import rebuild_index_version_task


class IndexVersionService:
    def __init__(
        self,
        *,
        session: AsyncSession,
        repository: IndexVersionRepository,
        knowledge_base_repository: KnowledgeBaseRepository,
        document_repository: DocumentRepository,
        indexing_service: IndexingService,
    ) -> None:
        self.session = session
        self.repository = repository
        self.knowledge_base_repository = knowledge_base_repository
        self.document_repository = document_repository
        self.indexing_service = indexing_service

    async def rebuild(
        self, *, kb_id: str, tenant_id: str, version: str = "v2"
    ) -> IndexVersionRebuildResponse:
        if version != "v2":
            raise_app_error(ErrorCode.PARAM_ERROR, "only v2 can be rebuilt")
        kb = await self.knowledge_base_repository.get_by_id(kb_id=kb_id, tenant_id=tenant_id)
        if kb is None:
            raise KnowledgeBaseNotFoundError(kb_id)
        versions = await self.repository.list(tenant_id=tenant_id, kb_id=kb_id)
        if any(item.status == IndexVersionStatus.BUILDING for item in versions):
            raise_app_error(ErrorCode.SYSTEM_BUSY, "an index rebuild is already running")
        existing = await self.repository.get(tenant_id=tenant_id, kb_id=kb_id, version=version)
        if existing is not None and existing.status == IndexVersionStatus.ACTIVE:
            raise_app_error(
                ErrorCode.PARAM_ERROR, "active index version cannot be rebuilt in place"
            )
        if existing is not None:
            await self._purge_version(kb_id=kb_id, tenant_id=tenant_id, version=version)
            await self.repository.delete(tenant_id=tenant_id, kb_id=kb_id, version=version)
        item = await self.repository.create(
            tenant_id=tenant_id,
            kb_id=kb_id,
            version=version,
            status=IndexVersionStatus.BUILDING,
        )
        await self.session.commit()
        try:
            rebuild_index_version_task.delay(tenant_id, kb_id, version)
        except Exception as exc:
            await self.session.rollback()
            item = await self.repository.get(tenant_id=tenant_id, kb_id=kb_id, version=version)
            if item is not None:
                await self.repository.mark_failed(item, str(exc))
                await self.session.commit()
            raise
        return IndexVersionRebuildResponse(kb_id=kb_id, version=version, status="building")

    async def list_versions(self, *, kb_id: str, tenant_id: str) -> IndexVersionListResponse:
        kb = await self.knowledge_base_repository.get_by_id(kb_id=kb_id, tenant_id=tenant_id)
        if kb is None:
            raise KnowledgeBaseNotFoundError(kb_id)
        items = await self.repository.list(tenant_id=tenant_id, kb_id=kb_id)
        return IndexVersionListResponse(
            kb_id=kb_id,
            active_index_version=getattr(kb, "active_index_version", "v1") or "v1",
            versions=[self._to_response(item) for item in items],
        )

    async def activate(self, *, kb_id: str, tenant_id: str, version: str) -> IndexVersionResponse:
        kb = await self.knowledge_base_repository.get_by_id(kb_id=kb_id, tenant_id=tenant_id)
        if kb is None:
            raise KnowledgeBaseNotFoundError(kb_id)
        item = await self.repository.get(tenant_id=tenant_id, kb_id=kb_id, version=version)
        if item is None:
            raise_app_error(ErrorCode.NOT_FOUND, "index version not found")
        try:
            activated = await self.repository.activate(
                tenant_id=tenant_id, kb_id=kb_id, version=version
            )
            kb.active_index_version = version
            await self.session.commit()
        except ValueError as exc:
            await self.session.rollback()
            raise_app_error(ErrorCode.PARAM_ERROR, str(exc))
        except Exception:
            await self.session.rollback()
            raise
        return self._to_response(activated)

    async def delete(self, *, kb_id: str, tenant_id: str, version: str) -> dict[str, str]:
        kb = await self.knowledge_base_repository.get_by_id(kb_id=kb_id, tenant_id=tenant_id)
        if kb is None:
            raise KnowledgeBaseNotFoundError(kb_id)
        if (getattr(kb, "active_index_version", "v1") or "v1") == version:
            raise_app_error(ErrorCode.PARAM_ERROR, "active index version cannot be deleted")
        item = await self.repository.get(tenant_id=tenant_id, kb_id=kb_id, version=version)
        if item is None:
            raise_app_error(ErrorCode.NOT_FOUND, "index version not found")
        await self._purge_version(kb_id=kb_id, tenant_id=tenant_id, version=version)
        await self.repository.delete(tenant_id=tenant_id, kb_id=kb_id, version=version)
        await self.session.commit()
        return {"kb_id": kb_id, "version": version}

    async def _purge_version(self, *, kb_id: str, tenant_id: str, version: str) -> None:
        documents = await self.document_repository.list_by_kb_id(kb_id=kb_id, tenant_id=tenant_id)
        for document in documents:
            await self.indexing_service.purge_document_chunks(document.id, index_version=version)

    @staticmethod
    def _to_response(item: Any) -> IndexVersionResponse:
        return IndexVersionResponse(
            version=item.version,
            status=item.status,
            document_count=item.document_count,
            chunk_count=item.chunk_count,
            error_message=item.error_message,
            created_at=item.created_at,
            completed_at=item.completed_at,
            activated_at=item.activated_at,
        )
