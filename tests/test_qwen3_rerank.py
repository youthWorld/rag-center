import json

import httpx
import pytest

from app.api.dependencies import get_rerank_provider
from app.core.config import Settings
from app.providers.rerank.qwen37 import Qwen37RerankProvider
from app.schemas.rag import RagRetrieveRequest


def _chunks(count=12):
    return [
        {
            "chunk_id": f"chunk-{i}",
            "document_id": "doc",
            "title": f"title {i}",
            "content": "content " + "x" * 1100,
            "score": i / 20,
            "metadata": {"private": "preserved"},
        }
        for i in range(count)
    ]


def _provider(handler):
    return Qwen37RerankProvider(
        base_url="https://workspace.cn-beijing.maas.aliyuncs.com/api/v1",
        api_key="test-secret",
        transport=httpx.MockTransport(handler),
    )


@pytest.mark.asyncio
async def test_native_request_and_index_mapping():
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        assert request.url.path.endswith("/api/v1/services/rerank/text-rerank/text-rerank")
        assert request.headers["Authorization"] == "Bearer test-secret"
        assert body["model"] == "qwen3.7-text-rerank"
        assert body["parameters"]["top_n"] == 10
        assert len(body["input"]["documents"]) == 12
        doc = body["input"]["documents"][0]
        assert doc.startswith("标题：title 0" + chr(10) + "正文：content ")
        assert len(doc.split("正文：")[1]) == 1024
        assert "private" not in doc
        return httpx.Response(
            200,
            json={
                "output": {
                    "results": [
                        {"index": i, "relevance_score": (10 if i == 11 else i) / 10}
                        for i in [11, 8, 7, 6, 5, 4, 3, 2, 1, 0]
                    ]
                }
            },
        )

    result = await _provider(handler).rerank(query="question", chunks=_chunks(), top_n=10)
    assert seen
    assert len(result) == 10
    assert result[0]["chunk_id"] == "chunk-11"
    assert result[0]["rerank_score"] == 1.0
    assert result[0]["metadata"] == {"private": "preserved"}


@pytest.mark.asyncio
async def test_fewer_candidates_and_stable_ties():
    def handler(request):
        assert json.loads(request.content)["parameters"]["top_n"] == 2
        return httpx.Response(
            200,
            json={
                "output": {
                    "results": [
                        {"index": 1, "relevance_score": 0.6},
                        {"index": 0, "relevance_score": 0.6},
                    ]
                }
            },
        )

    result = await _provider(handler).rerank(query="q", chunks=_chunks(2), top_n=10)
    assert [c["chunk_id"] for c in result] == ["chunk-0", "chunk-1"]


@pytest.mark.asyncio
async def test_empty_candidates_skip_remote_call():
    def handler(request):
        raise AssertionError("must not issue HTTP call")

    assert await _provider(handler).rerank(query="q", chunks=[], top_n=10) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "results",
    [
        None,
        [],
        [{"index": 2, "relevance_score": 0.4}],
        [{"index": 0, "relevance_score": 0.4}, {"index": 0, "relevance_score": 0.5}],
        [{"index": 0, "relevance_score": "NaN"}, {"index": 1, "relevance_score": 0.5}],
        [{"index": 0, "relevance_score": "not a number"}, {"index": 1, "relevance_score": 0.5}],
        [{"index": True, "relevance_score": 0.4}, {"index": 1, "relevance_score": 0.5}],
        [{"index": 0, "relevance_score": 0.4}, {"relevance_score": 0.5}],
    ],
)
async def test_invalid_results_are_rejected(results):
    with pytest.raises(ValueError):
        await _provider(
            lambda _: httpx.Response(200, json={"output": {"results": results}})
        ).rerank(
            query="q",
            chunks=_chunks(2),
            top_n=2,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 429, 500])
async def test_http_failure_has_redacted_error(status):
    provider = _provider(lambda _: httpx.Response(status, text="test-secret"))
    with pytest.raises(ValueError, match=f"rerank HTTP {status}") as exc:
        await provider.rerank(query="q", chunks=_chunks(2), top_n=2)
    assert "test-secret" not in str(exc.value)


@pytest.mark.asyncio
async def test_non_json_and_timeout():
    with pytest.raises(ValueError, match="rerank request failed"):
        await _provider(lambda _: httpx.Response(200, content=b"not json")).rerank(
            query="q",
            chunks=_chunks(2),
            top_n=2,
        )

    def timeout(_):
        raise httpx.ReadTimeout("test-secret")

    with pytest.raises(TimeoutError, match="rerank request timed out"):
        await _provider(timeout).rerank(query="q", chunks=_chunks(2), top_n=2)


def test_product_dependency_cannot_select_offline_llm_baseline():
    provider = get_rerank_provider(Settings(_env_file=None, rerank_provider="llm"))
    assert isinstance(provider, Qwen37RerankProvider)
    payload = RagRetrieveRequest.model_validate(
        {
            "kb_id": "kb",
            "user_id": "u",
            "query": "q",
            "profile": "custom",
            "offline_reranker": "llm",
        }
    )
    assert not hasattr(payload, "offline_reranker")
