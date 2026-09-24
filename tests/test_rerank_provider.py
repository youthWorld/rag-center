import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.config import Settings
from app.providers.llm.base import LLMProvider, LLMProviderError
from app.providers.llm.openai_compatible import OpenAICompatibleLLMProvider
from app.providers.rerank.noop import NoopRerankProvider
from eval.providers.llm_rerank_baseline import LLMRerankProvider


class FakeLLMProvider(LLMProvider):
    def __init__(self, response: dict) -> None:
        self.response = response
        self.calls: list[dict] = []

    async def chat_json(self, **kwargs) -> dict:
        self.calls.append(kwargs)
        return self.response


def _chunks() -> list[dict]:
    return [
        {
            "document_id": "doc-1",
            "chunk_id": "chunk-1",
            "title": "Refund policy",
            "content": "first content that is longer than the configured limit",
            "score": 0.86,
        },
        {
            "document_id": "doc-1",
            "chunk_id": "chunk-2",
            "title": "Refund policy",
            "content": "second content",
            "score": 0.73,
        },
    ]


@pytest.mark.asyncio
async def test_llm_rerank_sorts_chunks_and_limits_payload() -> None:
    llm = FakeLLMProvider(
        {
            "rankings": [
                {"chunk_id": "chunk-2", "rerank_score": 0.95},
                {"chunk_id": "unknown", "rerank_score": 1.0},
            ]
        }
    )
    provider = LLMRerankProvider(
        llm,
        max_candidates=2,
        chunk_max_chars=10,
        temperature=0.0,
        timeout_seconds=12,
    )

    result = await provider.rerank(query="question", chunks=_chunks(), top_n=2)

    assert [chunk["chunk_id"] for chunk in result] == ["chunk-2", "chunk-1"]
    assert result[0]["rerank_score"] == 0.95
    assert result[1]["rerank_score"] == 0.0
    call = llm.calls[0]
    assert call["temperature"] == 0.0
    assert call["timeout_seconds"] == 12
    assert call["user_payload"]["top_n"] == 2
    assert len(call["user_payload"]["candidates"]) == 2
    assert len(call["user_payload"]["candidates"][0]["content"]) == 10


@pytest.mark.asyncio
async def test_noop_rerank_preserves_vector_order() -> None:
    result = await NoopRerankProvider().rerank(
        query="question",
        chunks=_chunks(),
        top_n=1,
    )

    assert [chunk["chunk_id"] for chunk in result] == ["chunk-1"]
    assert result[0]["rerank_score"] is None


@pytest.mark.asyncio
async def test_openai_compatible_llm_provider_parses_json() -> None:
    provider = OpenAICompatibleLLMProvider(
        Settings(llm_api_key="test-key", llm_model="test-model")
    )
    create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"rankings": []}')
                )
            ]
        )
    )
    provider._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    result = await provider.chat_json(
        system_prompt="system",
        user_payload={"query": "question"},
        temperature=0.0,
        timeout_seconds=7,
    )

    assert result == {"rankings": []}
    kwargs = create.await_args.kwargs
    assert kwargs["model"] == "test-model"
    assert kwargs["messages"][0] == {"role": "system", "content": "system"}
    assert json.loads(kwargs["messages"][1]["content"]) == {"query": "question"}
    assert kwargs["response_format"] == {"type": "json_object"}
    assert kwargs["timeout"] == 7


@pytest.mark.asyncio
async def test_openai_compatible_llm_provider_rejects_invalid_json() -> None:
    provider = OpenAICompatibleLLMProvider(Settings(llm_api_key="test-key"))
    create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="not-json"))]
        )
    )
    provider._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    with pytest.raises(LLMProviderError, match="invalid JSON"):
        await provider.chat_json(system_prompt="system", user_payload={})
