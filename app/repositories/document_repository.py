from sqlalchemy.ext.asyncio import AsyncSession

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
