from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.providers.vectorstores.base import VectorStore
from app.repositories.chunk_repository import ChunkRepository


class PgVectorStore(VectorStore):
    records_request_attempts = True

    def __init__(self, session: AsyncSession, *, read_session_factory=None) -> None:
        self.repository = ChunkRepository(session)
        self.read_session_factory = read_session_factory

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
        return await self.similarity_search_scope(
            query_vector,
            tenant_id=tenant_id,
            index_versions={kb_id: index_version or "v1"},
            top_k=top_k,
        )

    async def similarity_search_scope(
        self,
        query_vector: list[float],
        *,
        tenant_id: str,
        index_versions: dict[str, str],
        top_k: int,
    ) -> list[dict[str, Any]]:
        if self.read_session_factory is None:
            raise RuntimeError("vector read session factory is required")
        async with self.read_session_factory() as session:
            try:
                await session.execute(text("SET TRANSACTION READ ONLY"))
                rows = await ChunkRepository(session).similarity_search_scope(
                    query_vector,
                    tenant_id=tenant_id,
                    index_versions=index_versions,
                    top_k=top_k,
                )
                return self._serialize(rows)
            finally:
                await session.rollback()

    @staticmethod
    def _serialize(rows) -> list[dict[str, Any]]:
        return [
            {
                "tenant_id": chunk.tenant_id,
                "kb_id": chunk.kb_id,
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
