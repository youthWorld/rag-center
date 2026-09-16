from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chunk import Chunk
from app.models.document import Document, DocumentStatus
from app.utils.id_generator import generate_id


class DocumentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self,
        *,
        tenant_id: str,
        kb_id: str,
        title: str,
        content: str,
        source_type: str = "text",
    ) -> Document:
        document = Document(
            id=generate_id(),
            tenant_id=tenant_id,
            kb_id=kb_id,
            title=title,
            source_type=source_type,
            content=content,
            status=int(DocumentStatus.PROCESSING),
        )
        self.session.add(document)
        await self.session.flush()
        return document

    async def get_by_id(
        self,
        document_id: str,
        tenant_id: str | None = None,
    ) -> Document | None:
        statement = select(Document).where(Document.id == document_id)
        if tenant_id is not None:
            statement = statement.where(Document.tenant_id == tenant_id)
        result = await self.session.execute(statement)
        return result.scalar_one_or_none()

    async def get_by_id_and_tenant(self, *, document_id: str, tenant_id: str) -> Document | None:
        return await self.get_by_id(document_id=document_id, tenant_id=tenant_id)

    async def list_by_kb_id(self, *, kb_id: str, tenant_id: str) -> list[Document]:
        statement = (
            select(Document)
            .where(Document.kb_id == kb_id, Document.tenant_id == tenant_id)
            .order_by(Document.created_at.asc(), Document.id.asc())
        )
        result = await self.session.execute(statement)
        return list(result.scalars().all())

    async def count_chunks(self, *, document_id: str) -> int:
        statement = select(func.count(Chunk.id)).where(Chunk.document_id == document_id)
        result = await self.session.execute(statement)
        return int(result.scalar_one())

    async def count_by_tenant_and_status(self, *, tenant_id: str, status: int) -> int:
        statement = select(func.count(Document.id)).where(
            Document.tenant_id == tenant_id,
            Document.status == status,
        )
        result = await self.session.execute(statement)
        return int(result.scalar_one())

    async def delete_by_id(self, *, document_id: str, tenant_id: str | None = None) -> None:
        statement = delete(Document).where(Document.id == document_id)
        if tenant_id is not None:
            statement = statement.where(Document.tenant_id == tenant_id)
        await self.session.execute(statement)
        await self.session.flush()

    async def delete_by_kb_id(self, *, kb_id: str, tenant_id: str) -> None:
        statement = delete(Document).where(
            Document.kb_id == kb_id,
            Document.tenant_id == tenant_id,
        )
        await self.session.execute(statement)
        await self.session.flush()
