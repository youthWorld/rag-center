from openai import AsyncOpenAI

from app.core.config import Settings
from app.core.error_codes import ErrorCode
from app.core.exceptions import (
    LLMServiceError,
    ServiceConfigurationError,
    map_llm_exception,
)
from app.core.logging import get_logger, log_llm_call
from app.providers.embedding.base import EmbeddingProvider


class OpenAICompatibleEmbeddingProvider(EmbeddingProvider):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: AsyncOpenAI | None = None
        self.logger = get_logger(__name__)

    def _get_client(self) -> AsyncOpenAI:
        if not self.settings.model_api_key:
            raise ServiceConfigurationError(
                internal_message="MODEL_API_KEY is not configured",
                context={"provider": type(self).__name__},
            )
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

    @staticmethod
    def _summarize_response(response) -> dict[str, int]:
        items = getattr(response, "data", [])
        dimensions = len(items[0].embedding) if items else 0
        return {"items": len(items), "dimensions": dimensions}

    async def _create_embeddings(self, input_value, *, operation: str):
        try:
            return await log_llm_call(
                lambda: self._get_client().embeddings.create(
                    model=self.settings.embedding_model,
                    input=input_value,
                    dimensions=self.settings.embedding_dimensions,
                ),
                model=self.settings.embedding_model,
                prompt=input_value,
                logger=self.logger,
                response_formatter=self._summarize_response,
            )
        except ServiceConfigurationError:
            raise
        except Exception as exception:
            raise map_llm_exception(
                exception,
                model=self.settings.embedding_model,
                operation=operation,
            ) from exception

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = await self._create_embeddings(texts, operation="embed_documents")
        if not response.data:
            raise LLMServiceError(
                code=ErrorCode.LLM_NO_RESPONSE,
                internal_message="embedding API returned no data",
                context={"operation": "embed_documents"},
            )
        return [
            self._validate_dimensions(list(item.embedding))
            for item in sorted(response.data, key=lambda item: item.index)
        ]

    async def embed_query(self, query: str) -> list[float]:
        response = await self._create_embeddings(query, operation="embed_query")
        if not response.data:
            raise LLMServiceError(
                code=ErrorCode.LLM_NO_RESPONSE,
                internal_message="embedding API returned no data",
                context={"operation": "embed_query"},
            )
        return self._validate_dimensions(list(response.data[0].embedding))
