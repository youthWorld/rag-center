from abc import ABC, abstractmethod
from typing import Any


class KeywordSearchProvider(ABC):
    async def keyword_search_scope(
        self,
        *,
        query: str,
        tenant_id: str,
        index_versions: dict[str, str],
        top_k: int = 20,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError("global keyword search is required")

    @abstractmethod
    async def add_chunks(self, chunks: list[dict[str, Any]]) -> None:
        pass

    @abstractmethod
    async def keyword_search(
        self,
        *,
        query: str,
        tenant_id: str,
        kb_id: str,
        top_k: int = 20,
        index_version: str | None = None,
    ) -> list[dict[str, Any]]:
        pass

    @abstractmethod
    async def delete_by_document_id(
        self, document_id: str, *, index_version: str | None = None
    ) -> None:
        pass
