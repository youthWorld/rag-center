import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from app.core.config import Settings
from app.core.http_clients import ExternalClients
from app.providers.embedding.openai_compatible import OpenAICompatibleEmbeddingProvider
from app.providers.keyword_search.elasticsearch import ElasticsearchKeywordSearchProvider
from app.providers.llm.openai_compatible import OpenAICompatibleLLMProvider
from app.providers.rerank.qwen37 import INSTRUCT, Qwen37RerankProvider
from app.providers.vectorstores.pgvector import PgVectorStore
from app.services.retrieval_usage import retrieval_requests


@pytest.mark.asyncio
async def test_model_pool_reuse_auth_timeout_and_sdk_borrowing():
    seen = []

    async def handler(request):
        seen.append(request)
        if request.url.path.endswith("embeddings"):
            return httpx.Response(
                200,
                json={
                    "data": [{"index": 0, "embedding": [1.0]}],
                    "model": "embedding",
                    "object": "list",
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                },
            )
        if "text-rerank" in request.url.path:
            assert json.loads(request.content)["parameters"]["instruct"] == INSTRUCT
            return httpx.Response(
                200, json={"output": {"results": [{"index": 0, "relevance_score": 0.9}]}}
            )
        return httpx.Response(
            200,
            json={
                "id": "c",
                "object": "chat.completion",
                "created": 0,
                "model": "chat",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "answer"},
                    }
                ],
            },
        )

    settings = Settings(
        hybrid_rrf_k=60,
        model_api_key="embedding-key",
        llm_api_key="chat-key",
        model_base_url="https://embedding.test/v1",
        llm_base_url="https://chat.test/v1",
        rerank_base_url="https://rerank.test/api/v1",
        rerank_api_key="rerank-key",
        embedding_dimensions=1,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        embedding = OpenAICompatibleEmbeddingProvider(settings, http_client=client)
        llm = OpenAICompatibleLLMProvider(settings, http_client=client)
        rerank = Qwen37RerankProvider.from_settings(settings, client=client)
        for _ in range(2):
            await embedding.embed_query("q")
            await llm.chat_text(system_prompt="system", user_payload={"q": "q"})
            await rerank.rerank(query="q", chunks=[{"content": "body"}], top_n=1)
        assert embedding._get_client()._client is client
        assert llm._get_client()._client is client
        assert embedding._get_client().max_retries == 2
        assert "authorization" not in client.headers
        assert [r.headers["authorization"] for r in seen] == [
            "Bearer embedding-key",
            "Bearer chat-key",
            "Bearer rerank-key",
        ] * 2
        assert seen[0].extensions["timeout"]["read"] == 600
        assert seen[2].extensions["timeout"]["read"] == settings.rerank_timeout_seconds
        await embedding._get_client().close()
        await llm._get_client().close()
        await rerank.close()
        assert not client.is_closed
    with pytest.raises(RuntimeError, match="closed"):
        await rerank.rerank(query="q", chunks=[{"content": "body"}], top_n=1)
    assert client.is_closed


@pytest.mark.asyncio
async def test_owner_closes_once_and_es_borrower_does_not_close():
    model = SimpleNamespace(aclose=AsyncMock())
    es = SimpleNamespace(close=AsyncMock())
    owner = ExternalClients(model, es)
    provider = ElasticsearchKeywordSearchProvider(Settings(hybrid_rrf_k=60), client=es)
    await provider.close()
    es.close.assert_not_awaited()
    await asyncio.gather(owner.aclose(), owner.aclose())
    model.aclose.assert_awaited_once()
    es.close.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body,failed",
    [
        ({"timed_out": True, "hits": {"hits": []}}, True),
        ({"_shards": {"failed": 1}, "hits": {"hits": []}}, True),
        ({"timed_out": False, "_shards": {"failed": 0}, "hits": {"hits": []}}, False),
    ],
)
async def test_es_partial_results_fail_but_complete_empty_results_succeed(body, failed):
    es = SimpleNamespace(
        indices=SimpleNamespace(exists=AsyncMock(return_value=True)),
        search=AsyncMock(return_value=body),
    )
    provider = ElasticsearchKeywordSearchProvider(Settings(hybrid_rrf_k=60), client=es)
    kwargs = dict(query="q", tenant_id="tenant", index_versions={"a": "v1", "b": "v2"}, top_k=20)
    if failed:
        with pytest.raises(RuntimeError, match="incomplete"):
            await provider.keyword_search_scope(**kwargs)
    else:
        assert await provider.keyword_search_scope(**kwargs) == []
    query = es.search.await_args.kwargs["query"]["bool"]
    assert query["filter"] == [{"term": {"tenant_id": "tenant"}}]
    assert query["minimum_should_match"] == 1
    for clause, kb, version, fields in zip(
        query["should"],
        ["a", "b"],
        ["v1", "v2"],
        [["title^2", "content"], ["retrieval_text", "title^2"]],
    ):
        assert clause["bool"]["filter"] == [
            {"term": {"kb_id": kb}},
            {"term": {"index_version": version}},
        ]
        assert clause["bool"]["must"]["multi_match"]["fields"] == fields


@pytest.mark.asyncio
async def test_es_exists_failure_does_not_count_as_search():
    es = SimpleNamespace(
        indices=SimpleNamespace(exists=AsyncMock(side_effect=RuntimeError("unavailable"))),
        search=AsyncMock(),
    )
    provider = ElasticsearchKeywordSearchProvider(Settings(hybrid_rrf_k=60), client=es)
    metrics = {"bm25_request_count": 0}
    token = retrieval_requests.set(metrics)
    try:
        with pytest.raises(RuntimeError):
            await provider.keyword_search_scope(
                query="q", tenant_id="t", index_versions={"a": "v1"}
            )
    finally:
        retrieval_requests.reset(token)
    assert metrics["bm25_request_count"] == 0
    es.search.assert_not_awaited()


@pytest.mark.asyncio
async def test_vector_sql_is_isolated_rolled_back_and_pair_filtered():
    statements = []

    class Session:
        rollback = AsyncMock()
        exited = False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            self.exited = True

        async def execute(self, statement):
            statements.append(statement)
            if str(statement) == "SET TRANSACTION READ ONLY":
                return None
            raise RuntimeError("transaction aborted")

    read_session = Session()
    request_session = SimpleNamespace(execute=AsyncMock())
    store = PgVectorStore(request_session, read_session_factory=lambda: read_session)
    with pytest.raises(RuntimeError):
        await store.similarity_search_scope(
            [1.0], tenant_id="tenant", index_versions={"a": "v1", "b": "v2"}, top_k=20
        )
    request_session.execute.assert_not_awaited()
    read_session.rollback.assert_awaited_once()
    assert read_session.exited
    assert str(statements[0]) == "SET TRANSACTION READ ONLY"
    sql = str(statements[1].compile(compile_kwargs={"literal_binds": True}))
    assert "chunks.kb_id = 'a' AND chunks.index_version = 'v1'" in sql
    assert "chunks.kb_id = 'b' AND chunks.index_version = 'v2'" in sql
    assert "chunks.tenant_id = 'tenant'" in sql
    assert "chunks.kb_id, chunks.id" in sql


@pytest.mark.asyncio
async def test_lifespan_owns_pools_once_and_dependencies_borrow(monkeypatch):
    from starlette.requests import Request

    from app.api.dependencies import get_external_clients
    from app.main import create_app

    owner = ExternalClients(SimpleNamespace(aclose=AsyncMock()), SimpleNamespace(close=AsyncMock()))
    monkeypatch.setattr("app.main.build_external_clients", lambda _: owner)
    app = create_app()
    async with app.router.lifespan_context(app):
        request = Request({"type": "http", "app": app})
        assert get_external_clients(request) is owner
        assert get_external_clients(request) is owner
        owner.model.aclose.assert_not_awaited()
    owner.model.aclose.assert_awaited_once()
    owner.elasticsearch.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_sdk_retry_counts_one_logical_embedding(monkeypatch):
    from app.services.retrieval_usage import retrieval_usage

    requests = []

    async def handler(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(503, json={"error": {"message": "temporary"}})
        return httpx.Response(
            200,
            json={
                "data": [{"index": 0, "embedding": [1.0]}],
                "model": "m",
                "object": "list",
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
        )

    metrics, calls = {}, {}
    token = retrieval_requests.set(metrics)
    call_token = retrieval_usage.set(calls)
    try:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider = OpenAICompatibleEmbeddingProvider(
                Settings(hybrid_rrf_k=60, model_api_key="test", embedding_dimensions=1),
                http_client=client,
            )
            monkeypatch.setattr(provider._get_client(), "_calculate_retry_timeout", lambda *args: 0)
            assert await provider.embed_query("q") == [1.0]
    finally:
        retrieval_requests.reset(token)
        retrieval_usage.reset(call_token)
    assert len(requests) == 2
    assert metrics == {"embedding_request_count": 1}
    assert calls == {"embedding": 1}
