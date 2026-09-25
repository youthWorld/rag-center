from __future__ import annotations

import asyncio
import sys
from collections.abc import Coroutine
from typing import Any, TypeVar

from app.celery_app import celery_app
from app.core.config import settings
from app.core.error_codes import ErrorCode
from app.core.exceptions import AppError
from app.db.session import session_factory
from app.models.document import DocumentStatus
from app.repositories.document_repository import DocumentRepository
from app.repositories.index_version_repository import IndexVersionRepository
from app.repositories.knowledge_base_repository import KnowledgeBaseRepository
from app.services.indexing_service import build_indexing_service
from app.services.reference_relation_service import ReferenceRelationService

_ResultT = TypeVar("_ResultT")


@celery_app.task(bind=True, max_retries=2, default_retry_delay=10)
def index_document_task(
    self,
    document_id: str,
    reparse: bool = False,
) -> dict[str, Any] | None:
    """Run indexing in a worker-owned event loop and database session."""

    try:
        # Celery invokes synchronous task functions, so each worker task owns an
        # isolated event loop for the async SQLAlchemy and provider calls.
        return _run_async(_index_document(document_id, reparse=reparse))
    except Exception as exc:
        if _is_retryable(exc) and self.request.retries < self.max_retries:
            _run_async(_reset_document_for_retry(document_id))
            raise self.retry(exc=exc) from exc
        raise


@celery_app.task(bind=True, max_retries=1, default_retry_delay=15)
def rebuild_index_version_task(
    self,
    tenant_id: str,
    kb_id: str,
    version: str = "v2",
) -> dict[str, Any]:
    """Build a side-by-side KB index version without replacing active chunks."""

    try:
        return _run_async(
            _rebuild_index_version(
                tenant_id=tenant_id,
                kb_id=kb_id,
                version=version,
            )
        )
    except Exception as exc:
        if _is_retryable(exc) and self.request.retries < self.max_retries:
            raise self.retry(exc=exc) from exc
        raise


async def _index_document(
    document_id: str,
    *,
    reparse: bool = False,
) -> dict[str, Any] | None:
    async with session_factory() as session:
        service = build_indexing_service(session, settings)
        try:
            result = await service.index_existing_document(document_id, reparse=reparse)
            if result is None:
                return None
            return {
                "document_id": document_id,
                "status": int(DocumentStatus.SUCCESS),
                "chunk_count": result,
            }
        finally:
            close = getattr(service.keyword_search_provider, "close", None)
            if close is not None:
                await close()


async def _reset_document_for_retry(document_id: str) -> None:
    async with session_factory() as session:
        document = await DocumentRepository(session).get_by_id(document_id=document_id)
        if document is None:
            return
        document.status = int(DocumentStatus.PROCESSING)
        document.error_message = None
        await session.commit()


async def _rebuild_index_version(*, tenant_id: str, kb_id: str, version: str) -> dict[str, Any]:
    async with session_factory() as session:
        service = build_indexing_service(session, settings)
        version_repository = IndexVersionRepository(session)
        document_repository = DocumentRepository(session)
        kb_repository = KnowledgeBaseRepository(session)
        item = await version_repository.get(
            tenant_id=tenant_id,
            kb_id=kb_id,
            version=version,
        )
        if item is None:
            raise LookupError("index version not found")
        try:
            knowledge_base = await kb_repository.get_by_id(
                kb_id=kb_id,
                tenant_id=tenant_id,
            )
            if knowledge_base is None:
                raise LookupError("knowledge base not found")
            documents = await document_repository.list_by_kb_id(
                kb_id=kb_id,
                tenant_id=tenant_id,
            )
            successful_documents = [
                document for document in documents if document.status == int(DocumentStatus.SUCCESS)
            ]
            for document in successful_documents:
                await service.purge_document_chunks(
                    document.id,
                    index_version=version,
                )
                await service.index_document(document, index_version=version)
            relation_stats = await ReferenceRelationService(session).build_relations(
                tenant_id=tenant_id,
                kb_id=kb_id,
                index_version=version,
            )
            document_count = len(successful_documents)
            chunk_count = await version_repository.count_chunks(
                tenant_id=tenant_id,
                kb_id=kb_id,
                version=version,
            )
            await version_repository.mark_ready(
                item,
                document_count=document_count,
                chunk_count=chunk_count,
            )
            await session.commit()
            return {
                "tenant_id": tenant_id,
                "kb_id": kb_id,
                "index_version": version,
                "status": "ready",
                "document_count": document_count,
                "chunk_count": chunk_count,
                **relation_stats,
            }
        except Exception as exc:
            await session.rollback()
            try:
                documents = await document_repository.list_by_kb_id(
                    kb_id=kb_id, tenant_id=tenant_id
                )
                for document in documents:
                    await service.purge_document_chunks(document.id, index_version=version)
                await session.commit()
            except Exception:
                await session.rollback()
                # Keep the version failed even if a remote index cannot be cleaned yet.
            item = await version_repository.get(
                tenant_id=tenant_id,
                kb_id=kb_id,
                version=version,
            )
            if item is not None:
                await version_repository.mark_failed(item, str(exc))
                await session.commit()
            raise
        finally:
            close = getattr(service.keyword_search_provider, "close", None)
            if close is not None:
                await close()


def _run_async(coro: Coroutine[Any, Any, _ResultT]) -> _ResultT:
    """Run one async task with an event loop compatible with psycopg on Windows."""

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    return asyncio.run(coro)


def _is_retryable(exception: Exception) -> bool:
    for candidate in _exception_chain(exception):
        if isinstance(candidate, AppError):
            if candidate.error_code in {
                ErrorCode.API_TIMEOUT,
                ErrorCode.API_RATE_LIMIT,
                ErrorCode.LLM_TIMEOUT,
                ErrorCode.LLM_RATE_LIMIT,
            }:
                return True
            continue

        if isinstance(candidate, (TimeoutError, ConnectionError, OSError)):
            return True

        exception_name = type(candidate).__name__.lower()
        if any(
            marker in exception_name
            for marker in ("timeout", "connection", "transport", "temporarily")
        ):
            return True
    return False


def _exception_chain(exception: Exception):
    current: BaseException | None = exception
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__
