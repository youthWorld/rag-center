from abc import ABC, abstractmethod
from typing import Any


class VectorStore(ABC):
    async def similarity_search_scope(
        self,
        query_vector: list[float],
        *,
        tenant_id: str,
        index_versions: dict[str, str],
        top_k: int,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError("global vector search is required")

    @abstractmethod
    async def add_chunks(self, chunks: list[dict[str, Any]]) -> None:
        pass

    @abstractmethod
    async def similarity_search(
        self,
        query_vector: list[float],
        *,
        tenant_id: str,
        kb_id: str,
        top_k: int = 5,
        index_version: str | None = None,
    ) -> list[dict[str, Any]]:
        pass

    @abstractmethod
    async def delete_by_document_id(
        self, document_id: str, *, index_version: str | None = None
    ) -> None:
        pass
