from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    @abstractmethod
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Create one embedding for each document chunk, preserving input order."""

    @abstractmethod
    async def embed_query(self, query: str) -> list[float]:
        """Create an embedding for a search query."""
