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


@pytest.mark.asyncio
async def test_embedding_provider_splits_document_batches() -> None:
    provider = OpenAICompatibleEmbeddingProvider(
        Settings(model_api_key="test-key", embedding_dimensions=3, embedding_batch_size=2)
    )

    async def create(**kwargs):
        texts = kwargs["input"]
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=index, embedding=[float(index), 0.0, 0.0])
                for index, _text in enumerate(texts)
            ]
        )

    create_mock = AsyncMock(side_effect=create)
    provider._client = SimpleNamespace(embeddings=SimpleNamespace(create=create_mock))

    embeddings = await provider.embed_documents(["one", "two", "three", "four", "five"])

    assert len(embeddings) == 5
    assert [call.kwargs["input"] for call in create_mock.await_args_list] == [
        ["one", "two"],
        ["three", "four"],
        ["five"],
    ]
    assert embeddings == [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]
