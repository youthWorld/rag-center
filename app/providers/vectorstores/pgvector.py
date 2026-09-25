from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.providers.vectorstores.base import VectorStore
from app.repositories.chunk_repository import ChunkRepository


class PgVectorStore(VectorStore):
    def __init__(self, session: AsyncSession) -> None:
        self.repository = ChunkRepository(session)

    async def add_chunks(self, chunks: list[dict[str, Any]]) -> None:
        await self.repository.add_chunks(chunks)

    async def similarity_search(
        self,
        query_vector: list[float],
        *,
        tenant_id: str,
        kb_id: str,
        top_k: int = 5,
        index_version: str | None = None,
    ) -> list[dict[str, Any]]:
        rows = await self.repository.similarity_search(
            query_vector,
            tenant_id=tenant_id,
            kb_id=kb_id,
            top_k=top_k,
            index_version=index_version,
        )
        return [
            {
                "document_id": chunk.document_id,
                "chunk_id": chunk.id,
                "title": chunk.title,
                "content": chunk.content,
                "index_version": chunk.index_version,
                "retrieval_text": chunk.retrieval_text,
                "section_id": chunk.section_id,
                "parent_section_id": chunk.parent_section_id,
                "order_index": chunk.order_index,
                "score": score,
                "metadata": dict(chunk.chunk_metadata or {}),
            }
            for chunk, score in rows
        ]

    async def delete_by_document_id(
        self, document_id: str, *, index_version: str | None = None
    ) -> None:
        await self.repository.delete_by_document_id(document_id, index_version=index_version)
