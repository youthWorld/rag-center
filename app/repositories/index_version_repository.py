from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chunk import Chunk
from app.models.document import Document
from app.models.index_version import IndexVersionStatus, KnowledgeBaseIndexVersion
from app.utils.id_generator import generate_id


class IndexVersionRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get(
        self, *, tenant_id: str, kb_id: str, version: str
    ) -> KnowledgeBaseIndexVersion | None:
        result = await self.session.execute(
            select(KnowledgeBaseIndexVersion).where(
                KnowledgeBaseIndexVersion.tenant_id == tenant_id,
                KnowledgeBaseIndexVersion.kb_id == kb_id,
                KnowledgeBaseIndexVersion.version == version,
            )
        )
        return result.scalar_one_or_none()

    async def list(self, *, tenant_id: str, kb_id: str) -> list[KnowledgeBaseIndexVersion]:
        result = await self.session.execute(
            select(KnowledgeBaseIndexVersion)
            .where(
                KnowledgeBaseIndexVersion.tenant_id == tenant_id,
                KnowledgeBaseIndexVersion.kb_id == kb_id,
            )
            .order_by(
                KnowledgeBaseIndexVersion.created_at.desc(),
                KnowledgeBaseIndexVersion.version.desc(),
            )
        )
        return list(result.scalars().all())

    async def create(
        self,
        *,
        tenant_id: str,
        kb_id: str,
        version: str,
        status: str = IndexVersionStatus.BUILDING,
    ) -> KnowledgeBaseIndexVersion:
        item = KnowledgeBaseIndexVersion(
            id=generate_id(),
            tenant_id=tenant_id,
            kb_id=kb_id,
            version=version,
            status=str(status),
        )
        self.session.add(item)
        await self.session.flush()
        return item

    async def count_documents(self, *, tenant_id: str, kb_id: str) -> int:
        result = await self.session.execute(
            select(func.count(Document.id)).where(
                Document.tenant_id == tenant_id,
                Document.kb_id == kb_id,
            )
        )
        return int(result.scalar_one())

    async def count_chunks(self, *, tenant_id: str, kb_id: str, version: str) -> int:
        result = await self.session.execute(
            select(func.count(Chunk.id)).where(
                Chunk.tenant_id == tenant_id,
                Chunk.kb_id == kb_id,
                Chunk.index_version == version,
            )
        )
        return int(result.scalar_one())

    async def mark_ready(
        self, item: KnowledgeBaseIndexVersion, *, document_count: int, chunk_count: int
    ) -> None:
        item.status = IndexVersionStatus.READY
        item.document_count = document_count
        item.chunk_count = chunk_count
        item.completed_at = datetime.now(UTC)
        item.error_message = None
        await self.session.flush()

    async def mark_failed(self, item: KnowledgeBaseIndexVersion, error_message: str) -> None:
        item.status = IndexVersionStatus.FAILED
        item.error_message = error_message[:2000]
        item.completed_at = datetime.now(UTC)
        await self.session.flush()

    async def activate(
        self, *, tenant_id: str, kb_id: str, version: str
    ) -> KnowledgeBaseIndexVersion:
        item = await self.get(tenant_id=tenant_id, kb_id=kb_id, version=version)
        if item is None:
            raise LookupError("index version not found")
        if item.status not in {IndexVersionStatus.READY, IndexVersionStatus.ACTIVE}:
            raise ValueError("only ready index versions can be activated")
        await self.session.execute(
            update(KnowledgeBaseIndexVersion)
            .where(
                KnowledgeBaseIndexVersion.tenant_id == tenant_id,
                KnowledgeBaseIndexVersion.kb_id == kb_id,
                KnowledgeBaseIndexVersion.status == IndexVersionStatus.ACTIVE,
                KnowledgeBaseIndexVersion.version != version,
            )
            .values(status=IndexVersionStatus.READY)
        )
        item.status = IndexVersionStatus.ACTIVE
        item.activated_at = datetime.now(UTC)
        await self.session.flush()
        return item

    async def delete(self, *, tenant_id: str, kb_id: str, version: str) -> None:
        await self.session.execute(
            delete(KnowledgeBaseIndexVersion).where(
                KnowledgeBaseIndexVersion.tenant_id == tenant_id,
                KnowledgeBaseIndexVersion.kb_id == kb_id,
                KnowledgeBaseIndexVersion.version == version,
            )
        )
        await self.session.flush()
