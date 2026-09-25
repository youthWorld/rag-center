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
    provider = OpenAICompatibleLLMProvider(Settings(llm_api_key="test-key", llm_model="test-model"))
    create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"rankings": []}'))]
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
async def test_openai_compatible_llm_provider_returns_portable_metadata() -> None:
    provider = OpenAICompatibleLLMProvider(
        Settings(llm_api_key="test-key", llm_model="request-model")
    )
    create = AsyncMock(
        return_value=SimpleNamespace(
            id="completion-id",
            _request_id="request-id",
            model="response-model",
            usage=SimpleNamespace(
                prompt_tokens=12,
                completion_tokens=5,
                total_tokens=17,
            ),
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content='{"rankings": []}'),
                )
            ],
        )
    )
    provider._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    result = await provider.chat_json_with_metadata(
        system_prompt="system", user_payload={"query": "question"}
    )

    assert result.output == {"rankings": []}
    assert result.metadata.request_model == "request-model"
    assert result.metadata.response_model == "response-model"
    assert result.metadata.request_id == "request-id"
    assert result.metadata.finish_reason == "stop"
    assert result.metadata.input_tokens == 12
    assert result.metadata.output_tokens == 5
    assert result.metadata.total_tokens == 17
    assert isinstance(result.metadata.latency_ms, int)


@pytest.mark.asyncio
async def test_openai_compatible_llm_provider_disables_qwen_thinking() -> None:
    provider = OpenAICompatibleLLMProvider(
        Settings(
            llm_api_key="test-key",
            llm_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            llm_model="qwen3.7-flash",
        )
    )
    create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))]
        )
    )
    provider._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    await provider.chat_json_with_metadata(
        system_prompt="system",
        user_payload={},
        enable_thinking=False,
    )

    assert create.await_args.kwargs["extra_body"] == {"enable_thinking": False}


@pytest.mark.asyncio
async def test_openai_compatible_llm_provider_omits_qwen_option_for_other_endpoints() -> None:
    provider = OpenAICompatibleLLMProvider(
        Settings(
            llm_api_key="test-key",
            llm_base_url="https://api.deepseek.com/v1",
            llm_model="deepseek-chat",
        )
    )
    create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))]
        )
    )
    provider._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    await provider.chat_json_with_metadata(
        system_prompt="system",
        user_payload={},
        enable_thinking=False,
    )

    assert "extra_body" not in create.await_args.kwargs


@pytest.mark.asyncio
async def test_openai_compatible_llm_provider_allows_missing_metadata_fields() -> None:
    provider = OpenAICompatibleLLMProvider(Settings(llm_api_key="test-key"))
    provider._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=AsyncMock(
                    return_value=SimpleNamespace(
                        choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))]
                    )
                )
            )
        )
    )

    result = await provider.chat_json_with_metadata(system_prompt="system", user_payload={})

    assert result.output == {}
    assert result.metadata.response_model is None
    assert result.metadata.request_id is None
    assert result.metadata.finish_reason is None
    assert result.metadata.input_tokens is None
    assert result.metadata.output_tokens is None
    assert result.metadata.total_tokens is None


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
