from abc import ABC, abstractmethod
from typing import Any


class RerankProvider(ABC):
    @abstractmethod
    async def rerank(
        self,
        *,
        query: str,
        chunks: list[dict[str, Any]],
        top_n: int,
    ) -> list[dict[str, Any]]:
        """Return reranked chunks while preserving their source fields."""
