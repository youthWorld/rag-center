from openai import AsyncOpenAI

from app.core.config import Settings
from app.providers.embedding.base import EmbeddingProvider


class OpenAICompatibleEmbeddingProvider(EmbeddingProvider):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: AsyncOpenAI | None = None

    def _get_client(self) -> AsyncOpenAI:
        if not self.settings.model_api_key:
            raise RuntimeError("MODEL_API_KEY is not configured")
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self.settings.model_api_key,
                base_url=self.settings.model_base_url,
            )
        return self._client

    def _validate_dimensions(self, embedding: list[float]) -> list[float]:
        actual_dimensions = len(embedding)
        expected_dimensions = self.settings.embedding_dimensions
        if actual_dimensions != expected_dimensions:
            raise RuntimeError(
                "embedding dimension mismatch: "
                f"configured {expected_dimensions}, model returned {actual_dimensions}"
            )
        return embedding

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = await self._get_client().embeddings.create(
            model=self.settings.embedding_model,
            input=texts,
            dimensions=self.settings.embedding_dimensions,
        )
        return [
            self._validate_dimensions(list(item.embedding))
            for item in sorted(response.data, key=lambda item: item.index)
        ]

    async def embed_query(self, query: str) -> list[float]:
        response = await self._get_client().embeddings.create(
            model=self.settings.embedding_model,
            input=query,
            dimensions=self.settings.embedding_dimensions,
        )
        return self._validate_dimensions(list(response.data[0].embedding))
