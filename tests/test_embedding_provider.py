from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.config import Settings
from app.providers.embedding.openai_compatible import OpenAICompatibleEmbeddingProvider


def test_embedding_provider_accepts_configured_dimension() -> None:
    provider = OpenAICompatibleEmbeddingProvider(
        Settings(model_api_key="test-key", embedding_dimensions=1536)
    )

    embedding = [0.0] * 1536

    assert provider._validate_dimensions(embedding) == embedding


def test_embedding_provider_rejects_unexpected_dimension() -> None:
    provider = OpenAICompatibleEmbeddingProvider(
        Settings(model_api_key="test-key", embedding_dimensions=1536)
    )

    with pytest.raises(
        RuntimeError,
        match="configured 1536, model returned 1024",
    ):
        provider._validate_dimensions([0.0] * 1024)


@pytest.mark.asyncio
async def test_embedding_request_includes_configured_dimension() -> None:
    provider = OpenAICompatibleEmbeddingProvider(
        Settings(model_api_key="test-key", embedding_dimensions=1536)
    )
    create = AsyncMock(
        return_value=SimpleNamespace(
            data=[SimpleNamespace(index=0, embedding=[0.0] * 1536)]
        )
    )
    provider._client = SimpleNamespace(embeddings=SimpleNamespace(create=create))

    await provider.embed_documents(["test document"])

    assert create.await_args.kwargs["dimensions"] == 1536
