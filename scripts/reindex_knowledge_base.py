from __future__ import annotations

import argparse
import asyncio
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import select

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import settings  # noqa: E402
from app.db.session import session_factory  # noqa: E402
from app.models.document import Document, DocumentStatus  # noqa: E402
from app.models.knowledge_base import KnowledgeBase  # noqa: E402
from app.repositories.document_repository import DocumentRepository  # noqa: E402
from app.services.indexing_service import (  # noqa: E402
    IndexingService,
    build_indexing_service,
)
from app.tasks.indexing import index_document_task  # noqa: E402


@dataclass(slots=True)
class ReindexStats:
    success: int = 0
    failed: int = 0
    skipped: int = 0
    failed_document_ids: list[str] | None = None

    def __post_init__(self) -> None:
        if self.failed_document_ids is None:
            self.failed_document_ids = []


class ReindexSelectionError(ValueError):
    """Raised when the requested knowledge base or documents are invalid."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reindex existing documents in one RAG Center knowledge base."
    )
    parser.add_argument(
        "--kb-id",
        required=True,
        help="Target knowledge base ID.",
    )
    parser.add_argument(
        "--tenant-id",
        help="Tenant ID. If omitted, it is read from the knowledge base.",
    )
    parser.add_argument(
        "--document-id",
        dest="document_ids",
        action="append",
        help="Only reindex this document. Repeat the option for multiple documents.",
    )
    return parser.parse_args(argv)


async def reindex_knowledge_base(
    kb_id: str,
    *,
    tenant_id: str | None = None,
    document_ids: Sequence[str] | None = None,
    session_factory_override: Any | None = None,
    task_override: Any | None = None,
    service_builder: Callable[[Any, Any], IndexingService] | None = None,
    app_settings: Any | None = None,
    wait_for_completion: bool = True,
    wait_timeout_seconds: int = 300,
    poll_interval_seconds: float = 1.0,
) -> ReindexStats:
    """Purge and enqueue reindex tasks for one knowledge base.

    The selection query runs once, while every document is processed in its own
    database session so one failed document cannot poison the remaining work.
    ``success`` means that the worker completed indexing successfully. A worker
    failure or a completion timeout is counted in ``failed``. The optional
    ``wait_for_completion`` switch is useful for unit tests; the CLI always
    waits for terminal document states.
    """

    db_session_factory = session_factory_override or session_factory
    task = task_override or index_document_task
    builder = service_builder or build_indexing_service
    configured_settings = app_settings or settings
    selected_ids, effective_tenant_id = await _select_document_ids(
        kb_id,
        tenant_id=tenant_id,
        document_ids=document_ids,
        db_session_factory=db_session_factory,
    )

    stats = ReindexStats()
    total = len(selected_ids)
    queued_ids: list[str] = []
    for position, document_id in enumerate(selected_ids, start=1):
        print(f"[{position}/{total}] processing document_id={document_id}")
        try:
            outcome = await _reindex_one_document(
                document_id,
                kb_id=kb_id,
                tenant_id=effective_tenant_id,
                db_session_factory=db_session_factory,
                task=task,
                service_builder=builder,
                app_settings=configured_settings,
            )
        except Exception as exc:
            stats.failed += 1
            stats.failed_document_ids.append(document_id)
            print(f"[{position}/{total}] failed document_id={document_id}: {exc}")
            continue

        if outcome == "skipped":
            stats.skipped += 1
            print(f"[{position}/{total}] skipped document_id={document_id} (PROCESSING)")
        else:
            queued_ids.append(document_id)
            print(f"[{position}/{total}] queued document_id={document_id}")

    if wait_for_completion and queued_ids:
        terminal_states = await _wait_for_documents(
            queued_ids,
            db_session_factory=db_session_factory,
            timeout_seconds=wait_timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )
        for document_id in queued_ids:
            status, error_message = terminal_states[document_id]
            if status == int(DocumentStatus.SUCCESS):
                stats.success += 1
                print(f"completed document_id={document_id} status=SUCCESS")
            else:
                stats.failed += 1
                stats.failed_document_ids.append(document_id)
                detail = f": {error_message}" if error_message else ""
                print(f"failed document_id={document_id} status={status}{detail}")
    else:
        stats.success = len(queued_ids)

    print(
        "reindex summary: "
        f"success={stats.success} failed={stats.failed} skipped={stats.skipped}"
    )
    return stats


async def _wait_for_documents(
    document_ids: Sequence[str],
    *,
    db_session_factory: Any,
    timeout_seconds: int,
    poll_interval_seconds: float,
) -> dict[str, tuple[int, str | None]]:
    if timeout_seconds <= 0:
        raise ValueError("wait timeout must be greater than zero")
    if poll_interval_seconds <= 0:
        raise ValueError("poll interval must be greater than zero")

    pending = set(document_ids)
    terminal_states: dict[str, tuple[int, str | None]] = {}
    deadline = time.monotonic() + timeout_seconds
    while pending and time.monotonic() < deadline:
        async with db_session_factory() as session:
            result = await session.execute(
                select(Document.id, Document.status, Document.error_message).where(
                    Document.id.in_(pending)
                )
            )
            rows = result.all()

        for document_id, status, error_message in rows:
            status_value = int(status)
            if status_value in {
                int(DocumentStatus.SUCCESS),
                int(DocumentStatus.FAILED),
            }:
                terminal_states[document_id] = (status_value, error_message)
                pending.discard(document_id)

        if pending:
            await asyncio.sleep(poll_interval_seconds)

    for document_id in pending:
        terminal_states[document_id] = (
            int(DocumentStatus.FAILED),
            f"timed out after {timeout_seconds}s waiting for Worker",
        )
    return terminal_states


async def _select_document_ids(
    kb_id: str,
    *,
    tenant_id: str | None,
    document_ids: Sequence[str] | None,
    db_session_factory: Any,
) -> tuple[list[str], str]:
    requested_ids = _unique_ids(document_ids)
    async with db_session_factory() as session:
        knowledge_base = await session.get(KnowledgeBase, kb_id)
        if knowledge_base is None:
            raise ReindexSelectionError(f"knowledge base does not exist: {kb_id}")
        if tenant_id is not None and knowledge_base.tenant_id != tenant_id:
            raise ReindexSelectionError(
                f"knowledge base {kb_id} does not belong to tenant {tenant_id}"
            )

        effective_tenant_id = tenant_id or knowledge_base.tenant_id
        if requested_ids:
            result = await session.execute(
                select(Document).where(Document.id.in_(requested_ids))
            )
            documents_by_id = {document.id: document for document in result.scalars().all()}
            invalid_ids = [
                document_id
                for document_id in requested_ids
                if (
                    document_id not in documents_by_id
                    or documents_by_id[document_id].kb_id != kb_id
                    or documents_by_id[document_id].tenant_id != effective_tenant_id
                )
            ]
            if invalid_ids:
                invalid = ", ".join(invalid_ids)
                raise ReindexSelectionError(
                    f"document does not belong to knowledge base {kb_id} and tenant "
                    f"{effective_tenant_id}: {invalid}"
                )
            return requested_ids, effective_tenant_id

        result = await session.execute(
            select(Document.id)
            .where(
                Document.kb_id == kb_id,
                Document.tenant_id == effective_tenant_id,
                Document.status == int(DocumentStatus.SUCCESS),
            )
            .order_by(Document.created_at.asc(), Document.id.asc())
        )
        return [document_id for document_id in result.scalars().all()], effective_tenant_id


async def _reindex_one_document(
    document_id: str,
    *,
    kb_id: str,
    tenant_id: str,
    db_session_factory: Any,
    task: Any,
    service_builder: Callable[[Any, Any], IndexingService],
    app_settings: Any,
) -> str:
    async with db_session_factory() as session:
        document = await DocumentRepository(session).get_by_id(document_id=document_id)
        if document is None:
            raise ReindexSelectionError(f"document does not exist: {document_id}")
        if document.kb_id != kb_id or document.tenant_id != tenant_id:
            raise ReindexSelectionError(
                f"document {document_id} does not belong to knowledge base {kb_id} "
                f"and tenant {tenant_id}"
            )
        if document.status == int(DocumentStatus.PROCESSING):
            return "skipped"
        if document.status not in {
            int(DocumentStatus.SUCCESS),
            int(DocumentStatus.FAILED),
        }:
            raise ReindexSelectionError(
                f"document {document_id} has unsupported status {document.status}"
            )

        service = service_builder(session, app_settings)
        try:
            await service.purge_document_chunks(document.id)
            document.status = int(DocumentStatus.PROCESSING)
            document.error_message = None
            await session.commit()
            try:
                task.delay(document.id)
            except Exception as exc:
                await service.mark_document_failed(document.id, str(exc))
                raise
        except Exception:
            if document.status != int(DocumentStatus.FAILED):
                await session.rollback()
            raise
        finally:
            keyword_provider = getattr(service, "keyword_search_provider", None)
            close = getattr(keyword_provider, "close", None)
            if close is not None:
                await close()
    return "queued"


def _unique_ids(document_ids: Sequence[str] | None) -> list[str]:
    return list(dict.fromkeys(document_ids or []))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        stats = asyncio.run(
            reindex_knowledge_base(
                args.kb_id,
                tenant_id=args.tenant_id,
                document_ids=args.document_ids,
            )
        )
    except ReindexSelectionError as exc:
        print(f"reindex aborted: {exc}", file=sys.stderr)
        return 2
    return 1 if stats.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
