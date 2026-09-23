import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.config import Settings
from app.providers.embedding.base import EmbeddingProvider
from app.providers.llm.base import LLMProvider
from app.providers.llm.openai_compatible import OpenAICompatibleLLMProvider
from app.providers.query.base import QueryContext, QueryProcessResult
from app.providers.query.llm_rewrite import (
    QUERY_REWRITE_SYSTEM_PROMPT,
    LLMRewriteProcessor,
)
from app.providers.query.pipeline import QueryPipeline
from app.providers.query.synonym_expander import SynonymExpander
from app.providers.rerank.base import RerankProvider
from app.providers.vectorstores.base import VectorStore
from app.schemas.rag import RagRetrieveRequest
from app.services.rag_service import RagService


class FakeLLMProvider(LLMProvider):
    def __init__(self, response: dict | None = None, *, error: Exception | None = None) -> None:
        self.response = response or {}
        self.error = error
        self.calls: list[dict] = []

    async def chat_json(self, **kwargs) -> dict:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class FakeEmbeddingProvider(EmbeddingProvider):
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    async def embed_query(self, query: str) -> list[float]:
        self.queries.append(query)
        return [1.0]


class FakeVectorStore(VectorStore):
    def __init__(self) -> None:
        self.queries: list[dict] = []

    async def add_chunks(self, chunks: list[dict]) -> None:
        del chunks

    async def similarity_search(self, query_vector, *, tenant_id, kb_id, top_k=5):
        self.queries.append(
            {
                "query_vector": query_vector,
                "tenant_id": tenant_id,
                "kb_id": kb_id,
                "top_k": top_k,
            }
        )
        return [
            {
                "document_id": "doc-1",
                "chunk_id": "chunk-1",
                "title": "Background check",
                "content": "content",
                "score": 0.9,
            }
        ]

    async def delete_by_document_id(self, document_id: str) -> None:
        del document_id


class FakeRerankProvider(RerankProvider):
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def rerank(self, *, query: str, chunks: list[dict], top_n: int) -> list[dict]:
        self.queries.append(query)
        return chunks[:top_n]


@pytest.mark.asyncio
async def test_synonym_expander_matches_terms_and_deduplicates_expansions() -> None:
    result = await SynonymExpander().process(
        QueryProcessResult(
            raw_query="请告诉我背调和工资要求",
            effective_query="请告诉我背调和工资要求",
            search_query="请告诉我背调和工资要求",
        ),
        context=QueryContext(
            name="招聘知识库",
            settings={
                "synonyms": [
                    {"terms": ["背调"], "expand": ["背景调查", "标准问题清单"]},
                    {"terms": ["工资"], "expand": ["职级", "背景调查"]},
                ]
            },
        ),
    )

    assert result.synonym_applied is True
    assert result.synonym_expansions == ["背景调查", "标准问题清单", "职级"]
    assert result.search_query == "请告诉我背调和工资要求 背景调查 标准问题清单 职级"


@pytest.mark.asyncio
async def test_synonym_expander_matches_english_terms_case_insensitively() -> None:
    search_query, applied, expansions = SynonymExpander.expand(
        "How do I use LangGraph?",
        {"synonyms": [{"terms": ["langgraph"], "expand": ["state graph"]}]},
    )

    assert applied is True
    assert expansions == ["state graph"]
    assert search_query.endswith(" state graph")


@pytest.mark.asyncio
async def test_llm_rewrite_uses_description_and_rewrite_hint() -> None:
    llm = FakeLLMProvider({"query": "背景调查标准问题有哪些"})
    processor = LLMRewriteProcessor(llm, timeout_ms=2000)

    result = await processor.process(
        QueryProcessResult(
            raw_query="背调要问啥",
            effective_query="背调要问啥",
            search_query="背调要问啥",
        ),
        context=QueryContext(
            name="客服招聘知识库",
            description="招聘流程和面试标准",
            settings={"rewrite_hint": "背调指背景调查"},
        ),
    )

    assert result.effective_query == "背景调查标准问题有哪些"
    assert result.search_query == result.effective_query
    assert result.strategy == "rewrite"
    assert result.degraded is False
    assert result.rewrite_latency_ms >= 0
    assert result.application_model_calls == 1
    call = llm.calls[0]
    assert call["system_prompt"] == QUERY_REWRITE_SYSTEM_PROMPT
    assert call["user_payload"] == {
        "query": "背调要问啥",
        "kb_name": "客服招聘知识库",
        "kb_description": "招聘流程和面试标准\n背调指背景调查",
    }
    assert call["temperature"] == 0.0
    assert call["timeout_seconds"] == 2.0
    assert call["max_tokens"] == 128


@pytest.mark.asyncio
async def test_llm_rewrite_degrades_without_breaking_query_processing() -> None:
    processor = LLMRewriteProcessor(
        FakeLLMProvider(error=RuntimeError("provider unavailable")),
        timeout_ms=2000,
    )

    result = await processor.process(
        QueryProcessResult(
            raw_query="背调要问啥",
            effective_query="背调要问啥",
            search_query="背调要问啥",
        ),
        context=QueryContext(name="招聘知识库"),
    )

    assert result.strategy == "rewrite"
    assert result.degraded is True
    assert result.effective_query == "背调要问啥"
    assert result.search_query == "背调要问啥"
    assert result.degraded_reason == "provider unavailable"
    assert result.application_model_calls == 1


@pytest.mark.asyncio
async def test_openai_compatible_llm_provider_supports_plain_text_completion() -> None:
    provider = OpenAICompatibleLLMProvider(
        Settings(llm_api_key="test-key", llm_model="test-model")
    )
    create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="背景调查标准问题有哪些")
                )
            ]
        )
    )
    provider._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    result = await provider.chat_text(
        system_prompt="system",
        user_payload={"query": "question"},
        temperature=0.0,
        timeout_seconds=2.0,
        max_tokens=128,
    )

    assert result == "背景调查标准问题有哪些"
    kwargs = create.await_args.kwargs
    assert kwargs["model"] == "test-model"
    assert kwargs["temperature"] == 0.0
    assert kwargs["max_tokens"] == 128
    assert kwargs["timeout"] == 2.0
    assert "response_format" not in kwargs


@pytest.mark.asyncio
async def test_llm_rewrite_timeout_degrades() -> None:
    class SlowLLMProvider(FakeLLMProvider):
        async def chat_text(self, **kwargs) -> str:
            del kwargs
            await asyncio.sleep(0.05)
            return "不会返回"

    processor = LLMRewriteProcessor(SlowLLMProvider(), timeout_ms=1)
    result = await processor.process(
        QueryProcessResult("query", "query", "query"),
        context=QueryContext(name="KB"),
    )

    assert result.degraded is True
    assert result.degraded_reason == "query rewrite timed out"
    assert result.effective_query == "query"


@pytest.mark.asyncio
async def test_pipeline_request_options_override_global_rewrite_setting() -> None:
    llm = FakeLLMProvider({"query": "rewritten"})
    pipeline = QueryPipeline(
        rewrite_enabled=True,
        rewrite_processor=LLMRewriteProcessor(llm),
    )
    knowledge_base = SimpleNamespace(
        name="KB",
        description="domain",
        settings={"synonyms": [{"terms": ["raw"], "expand": ["expanded"]}]},
    )

    disabled = await pipeline.process(
        "raw",
        knowledge_base=knowledge_base,
        query_options={"enabled": False},
    )
    enabled = await pipeline.process(
        "raw",
        knowledge_base=knowledge_base,
        query_options={"enabled": True, "strategy": "rewrite"},
    )

    assert disabled.strategy == "noop"
    assert disabled.degraded is False
    assert disabled.search_query == "raw expanded"
    assert enabled.strategy == "rewrite"
    assert enabled.effective_query == "rewritten"
    assert enabled.search_query == "rewritten"
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_explicit_enabled_wins_over_conflicting_strategy_in_service_and_pipeline() -> None:
    service = object.__new__(RagService)
    service.settings = Settings(query_rewrite_enabled=False)
    llm = FakeLLMProvider({"query": "rewritten"})
    pipeline = QueryPipeline(
        rewrite_enabled=False,
        rewrite_processor=LLMRewriteProcessor(llm),
    )
    knowledge_base = SimpleNamespace(name="KB", settings={})

    for enabled, strategy, expected_strategy in (
        (True, "noop", "rewrite"),
        (False, "rewrite", "noop"),
    ):
        options = {"enabled": enabled, "strategy": strategy}
        request = RagRetrieveRequest(
            kb_id="kb-test", user_id="user-test", query="raw", query_options=options
        )
        result = await pipeline.process(
            "raw", knowledge_base=knowledge_base, query_options=request.query_options
        )
        assert service._resolve_query_rewrite_enabled(request) is enabled
        assert result.strategy == expected_strategy

    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_pipeline_can_disable_synonym_expansion_per_request() -> None:
    pipeline = QueryPipeline()
    knowledge_base = SimpleNamespace(
        name="KB",
        description=None,
        settings={"synonyms": [{"terms": ["退款"], "expand": ["原路退回"]}]},
    )

    result = await pipeline.process(
        "退款",
        knowledge_base=knowledge_base,
        query_options={
            "enabled": False,
            "strategy": "noop",
            "synonym_enabled": False,
        },
    )

    assert result.search_query == "退款"
    assert result.synonym_enabled is False
    assert result.synonym_applied is False


@pytest.mark.asyncio
async def test_synonyms_are_scoped_to_each_knowledge_base() -> None:
    pipeline = QueryPipeline()
    kb_one = SimpleNamespace(
        name="KB one",
        description=None,
        settings={"synonyms": [{"terms": ["status"], "expand": ["state"]}]},
    )
    kb_two = SimpleNamespace(
        name="KB two",
        description=None,
        settings={"synonyms": [{"terms": ["status"], "expand": ["lifecycle"]}]},
    )

    first = await pipeline.process("status", knowledge_base=kb_one)
    second = await pipeline.process("status", knowledge_base=kb_two)

    assert first.search_query == "status state"
    assert second.search_query == "status lifecycle"


@pytest.mark.asyncio
async def test_rag_service_uses_processed_query_for_retrieval_but_raw_query_for_rerank() -> None:
    llm = FakeLLMProvider({"query": "背景调查标准问题有哪些"})
    embedding = FakeEmbeddingProvider()
    vector_store = FakeVectorStore()
    rerank = FakeRerankProvider()
    log_repository = SimpleNamespace(create=AsyncMock())
    pipeline = QueryPipeline(
        rewrite_enabled=False,
        rewrite_processor=LLMRewriteProcessor(llm),
    )
    knowledge_base = SimpleNamespace(
        id="kb-test",
        name="招聘知识库",
        description="招聘制度",
        settings={"synonyms": [{"terms": ["背景调查"], "expand": ["标准问题清单"]}]},
    )
    service = RagService(
        session=SimpleNamespace(commit=AsyncMock()),
        settings=Settings(top_k=1, query_rewrite_enabled=False),
        knowledge_base_repository=SimpleNamespace(
            get_by_id=AsyncMock(return_value=knowledge_base)
        ),
        retrieval_log_repository=log_repository,
        embedding_provider=embedding,
        vector_store=vector_store,
        rerank_provider=rerank,
        query_pipeline=pipeline,
    )

    response = await service.retrieve(
        RagRetrieveRequest(
            kb_id="kb-test",
            user_id="user-test",
            query="背调要问啥",
            query_options={"enabled": True, "strategy": "rewrite"},
            rerank_options={"enabled": True, "top_n": 1},
        ),
        tenant_id="tenant-test",
    )

    assert embedding.queries == ["背景调查标准问题有哪些 标准问题清单"]
    assert response.query == "背调要问啥"
    assert rerank.queries == ["背调要问啥"]
    assert response.metadata["query_processing"] == {
        "raw_query": "背调要问啥",
        "effective_query": "背景调查标准问题有哪些",
        "search_query": "背景调查标准问题有哪些 标准问题清单",
        "strategy": "rewrite",
        "rewrite_latency_ms": response.metadata["query_processing"]["rewrite_latency_ms"],
        "degraded": False,
        "degraded_reason": None,
        "synonym_enabled": None,
        "synonym_applied": True,
        "synonym_expansions": ["标准问题清单"],
        "application_model_calls": 1,
    }
    assert response.metadata["application_model_calls"] == 2
    assert response.metadata["application_model_call_details"] == {
        "query_rewrite": 1,
        "rerank": 1,
    }


@pytest.mark.asyncio
async def test_rag_service_keeps_query_processing_null_for_legacy_request() -> None:
    service = RagService(
        session=SimpleNamespace(commit=AsyncMock()),
        settings=Settings(top_k=1),
        knowledge_base_repository=SimpleNamespace(
            get_by_id=AsyncMock(return_value=SimpleNamespace(id="kb-test"))
        ),
        retrieval_log_repository=SimpleNamespace(create=AsyncMock()),
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=FakeVectorStore(),
    )

    response = await service.retrieve(
        RagRetrieveRequest(kb_id="kb-test", user_id="user-test", query="question"),
        tenant_id="tenant-test",
    )

    assert response.metadata["query_processing"] is None
