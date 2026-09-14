from abc import ABC, abstractmethod
from typing import Any


class VectorStore(ABC):
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
    ) -> list[dict[str, Any]]:
        pass

    @abstractmethod
    async def delete_by_document_id(self, document_id: str) -> None:
        pass
