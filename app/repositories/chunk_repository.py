from collections.abc import Sequence
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chunk import Chunk
from app.models.document import Document, DocumentStatus


class ChunkRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add_chunks(self, chunks: Sequence[dict[str, Any]]) -> None:
        self.session.add_all(
            [
                Chunk(
                    id=chunk["id"],
                    tenant_id=chunk["tenant_id"],
                    kb_id=chunk["kb_id"],
                    document_id=chunk["document_id"],
                    title=chunk["title"],
                    content=chunk["content"],
                    index_version=str(chunk.get("index_version") or "v1"),
                    retrieval_text=chunk.get("retrieval_text"),
                    section_id=chunk.get("section_id"),
                    parent_section_id=chunk.get("parent_section_id"),
                    order_index=chunk.get("order_index"),
                    chunk_metadata=chunk.get("metadata", {}),
                    embedding=chunk["embedding"],
                )
                for chunk in chunks
            ]
        )
        await self.session.flush()

    async def similarity_search(
        self,
        query_vector: list[float],
        *,
        tenant_id: str,
        kb_id: str,
        top_k: int,
        index_version: str | None = None,
    ) -> list[tuple[Chunk, float]]:
        distance = Chunk.embedding.cosine_distance(query_vector)
        statement = (
            select(Chunk, (1 - distance).label("score"))
            .join(Document, Chunk.document_id == Document.id)
            .where(
                Chunk.tenant_id == tenant_id,
                Chunk.kb_id == kb_id,
                Document.status == int(DocumentStatus.SUCCESS),
            )
            .order_by(distance)
            .limit(top_k)
        )
        if index_version is not None:
            statement = statement.where(Chunk.index_version == index_version)
        result = await self.session.execute(statement)
        return [(chunk, float(score)) for chunk, score in result.all()]

    async def delete_by_document_id(
        self, document_id: str, *, index_version: str | None = None
    ) -> None:
        statement = delete(Chunk).where(Chunk.document_id == document_id)
        if index_version is not None:
            statement = statement.where(Chunk.index_version == index_version)
        await self.session.execute(statement)
        await self.session.flush()
