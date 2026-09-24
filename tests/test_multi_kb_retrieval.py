import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.core.error_codes import ErrorCode
from app.core.exceptions import AppError
from app.providers.embedding.base import EmbeddingProvider
from app.providers.query.pipeline import QueryPipeline
from app.providers.vectorstores.base import VectorStore
from app.schemas.rag import RagRetrieveRequest
from app.services.multi_kb_fusion_service import MultiKBFusionService
from app.services.rag_service import RagService
from app.tenant.plan_resolver import PlanResolver


def _chunk(chunk_id: str, *, score: float) -> dict:
    return {
        "document_id": f"document-{chunk_id}",
        "chunk_id": chunk_id,
        "title": f"Title {chunk_id}",
        "content": f"Content {chunk_id}",
        "score": score,
    }


def test_multi_kb_fusion_sums_rrf_scores_for_duplicate_chunks() -> None:
    service = MultiKBFusionService(rrf_k=60)

    result = service.fuse(
        {
            "kb-a": [_chunk("shared", score=0.9), _chunk("a-only", score=0.8)],
            "kb-b": [_chunk("b-only", score=0.9), _chunk("shared", score=0.8)],
        }
    )

    assert [chunk["chunk_id"] for chunk in result] == ["shared", "b-only", "a-only"]
    assert result[0]["score"] == pytest.approx(1 / 61 + 1 / 62)
    assert result[0]["kb_id"] == "kb-a"


class FakeEmbeddingProvider(EmbeddingProvider):
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    async def embed_query(self, query: str) -> list[float]:
        self.calls.append(query)
        return [1.0]


class FakeVectorStore(VectorStore):
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.active = 0
        self.max_active = 0

    async def add_chunks(self, chunks: list[dict]) -> None:
        del chunks

    async def similarity_search(self, query_vector, *, tenant_id, kb_id, top_k=5):
        self.calls.append(
            {
                "query_vector": query_vector,
                "tenant_id": tenant_id,
                "kb_id": kb_id,
                "top_k": top_k,
            }
        )
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0.01)
            return {
                "kb-a": [_chunk("shared", score=0.9), _chunk("a-only", score=0.8)],
                "kb-b": [_chunk("b-only", score=0.9), _chunk("shared", score=0.8)],
            }[kb_id][:top_k]
        finally:
            self.active -= 1

    async def delete_by_document_id(self, document_id: str) -> None:
        del document_id


class FakeKnowledgeBaseRepository:
    def __init__(self, available_ids: set[str] | None = None) -> None:
        self.available_ids = available_ids or {"kb-a", "kb-b"}
        self.get_by_ids_calls: list[dict] = []

    async def get_by_ids(self, *, kb_ids: list[str], tenant_id: str):
        self.get_by_ids_calls.append({"kb_ids": kb_ids, "tenant_id": tenant_id})
        return [
            SimpleNamespace(id=kb_id, name=f"Name {kb_id}")
            for kb_id in kb_ids
            if kb_id in self.available_ids
        ]


class FakeTenantRepository:
    def __init__(self, plan: str) -> None:
        self.tenant = SimpleNamespace(id="tenant-test", plan=plan)

    async def get_by_id(self, tenant_id: str):
        assert tenant_id == self.tenant.id
        return self.tenant


class CountingQueryPipeline(QueryPipeline):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict] = []

    async def process(self, raw_query: str, *, knowledge_base, query_options=None):
        self.calls.append({"query": raw_query, "knowledge_base": knowledge_base.id})
        return await super().process(
            raw_query,
            knowledge_base=knowledge_base,
            query_options=query_options,
        )


class FakeRateLimitService:
    def __init__(self) -> None:
        self.check_calls: list[str] = []
        self.record_calls: list[str] = []

    async def check_retrieve(self, tenant_id: str, plan) -> None:
        del plan
        self.check_calls.append(tenant_id)

    async def record_retrieve_success(self, tenant_id: str) -> None:
        self.record_calls.append(tenant_id)


def _service(
    *,
    plan: str = "pro",
    available_ids: set[str] | None = None,
):
    embedding = FakeEmbeddingProvider()
    vector_store = FakeVectorStore()
    knowledge_base_repository = FakeKnowledgeBaseRepository(available_ids)
    log_repository = SimpleNamespace(create=AsyncMock(return_value=SimpleNamespace(id="log-test")))
    rate_limit_service = FakeRateLimitService()
    query_pipeline = CountingQueryPipeline()
    service = RagService(
        session=SimpleNamespace(commit=AsyncMock()),
        settings=Settings(top_k=5, retrieval_mode="vector"),
        knowledge_base_repository=knowledge_base_repository,
        retrieval_log_repository=log_repository,
        embedding_provider=embedding,
        vector_store=vector_store,
        query_pipeline=query_pipeline,
        plan_resolver=PlanResolver(FakeTenantRepository(plan)),
        rate_limit_service=rate_limit_service,
    )
    return (
        service,
        embedding,
        vector_store,
        knowledge_base_repository,
        log_repository,
        rate_limit_service,
        query_pipeline,
    )


@pytest.mark.asyncio
async def test_multi_kb_quality_rerank_preserves_sources_and_falls_back() -> None:
    service, _, _, _, _, _, _ = _service()
    service.keyword_search_provider = SimpleNamespace(keyword_search=AsyncMock(return_value=[]))

    class CapturingReranker:
        def __init__(self):
            self.seen = None
            self.fail = False

        async def rerank(self, *, query, chunks, top_n):
            self.seen = list(chunks)
            if self.fail:
                raise RuntimeError("confidential")
            return [{**chunk, "rerank_score": 0.7} for chunk in reversed(chunks)][:top_n]

    reranker = CapturingReranker()
    service.rerank_provider = reranker
    request = RagRetrieveRequest(
        kb_ids=["kb-a", "kb-b"], user_id="multi-quality", query="refund", profile="quality"
    )
    result = await service.retrieve(request, tenant_id="tenant-test")
    assert len(reranker.seen) == 3
    assert result.metadata["rerank"]["candidate_count"] == 3
    assert result.metadata["rerank"]["returned_count"] == 3
    assert result.metadata["rerank"]["top_n"] == 10
    assert [chunk.chunk_id for chunk in result.retrieved_chunks] == [
        chunk["chunk_id"] for chunk in reversed(reranker.seen)
    ]
    assert {chunk.kb_id for chunk in result.retrieved_chunks} == {"kb-a", "kb-b"}
    reranker.fail = True
    fallback = await service.retrieve(request, tenant_id="tenant-test")
    assert [chunk.chunk_id for chunk in fallback.retrieved_chunks] == [
        chunk["chunk_id"] for chunk in reranker.seen
    ]
    assert all(chunk.rerank_score is None for chunk in fallback.retrieved_chunks)
    assert fallback.metadata["rerank"]["degraded"] is True
    assert "confidential" not in fallback.metadata["rerank"]["error"]


@pytest.mark.asyncio
async def test_multi_kb_retrieve_processes_query_once_and_fuses_parallel_candidates() -> None:
    (
        service,
        embedding,
        vector_store,
        knowledge_base_repository,
        log_repository,
        rate_limit_service,
        query_pipeline,
    ) = _service()

    response = await service.retrieve(
        RagRetrieveRequest(
            kb_ids=["kb-a", "kb-b"],
            user_id="user-test",
            query="refund",
            profile="custom",
            top_k=3,
        ),
        tenant_id="tenant-test",
    )

    assert embedding.calls == ["refund"]
    assert query_pipeline.calls == [{"query": "refund", "knowledge_base": "kb-a"}]
    assert vector_store.max_active == 2
    assert {call["kb_id"] for call in vector_store.calls} == {"kb-a", "kb-b"}
    assert {call["top_k"] for call in vector_store.calls} == {10}
    assert knowledge_base_repository.get_by_ids_calls == [
        {"kb_ids": ["kb-a", "kb-b"], "tenant_id": "tenant-test"}
    ]
    assert [chunk.chunk_id for chunk in response.retrieved_chunks] == [
        "shared",
        "b-only",
        "a-only",
    ]
    assert {chunk.kb_id for chunk in response.retrieved_chunks} == {"kb-a", "kb-b"}
    assert response.kb_id == "kb-a"
    assert response.kb_ids == ["kb-a", "kb-b"]
    assert response.metadata["retrieval"] == {
        "mode": "vector",
        "fusion": "rrf",
        "rrf_k": 60,
        "vector_store": "pgvector",
        "keyword_search": None,
        "vector_top_k": 10,
        "bm25_top_k": 0,
        "vector_count": 4,
        "bm25_count": 0,
        "fused_count": 3,
        "multi_kb": True,
        "kb_count": 2,
        "per_kb_top_k": 10,
    }
    assert log_repository.create.await_args.kwargs["kb_id"] == "kb-a"
    assert log_repository.create.await_args.kwargs["kb_ids"] == ["kb-a", "kb-b"]
    assert rate_limit_service.check_calls == ["tenant-test"]
    assert rate_limit_service.record_calls == ["tenant-test"]


@pytest.mark.asyncio
async def test_multi_kb_retrieve_rejects_missing_or_cross_tenant_knowledge_base() -> None:
    service, _, vector_store, _, _, _, _ = _service(available_ids={"kb-a"})

    with pytest.raises(AppError) as raised:
        await service.retrieve(
            RagRetrieveRequest(
                kb_ids=["kb-a", "kb-missing"],
                user_id="user-test",
                query="refund",
                profile="custom",
            ),
            tenant_id="tenant-test",
        )

    assert raised.value.code == ErrorCode.NOT_FOUND.code
    assert raised.value.context == {"missing_kb_id": "kb-missing"}
    assert not vector_store.calls


@pytest.mark.asyncio
async def test_free_plan_rejects_multi_kb_before_search() -> None:
    service, _, vector_store, _, _, _, _ = _service(plan="free")

    with pytest.raises(AppError) as raised:
        await service.retrieve(
            RagRetrieveRequest(
                kb_ids=["kb-a", "kb-b"],
                user_id="user-test",
                query="refund",
                profile="speed",
            ),
            tenant_id="tenant-test",
        )

    assert raised.value.code == ErrorCode.FEATURE_NOT_ALLOWED.code == 20013
    assert not vector_store.calls


def test_retrieve_request_requires_a_knowledge_base_and_rejects_empty_lists() -> None:
    with pytest.raises(ValidationError):
        RagRetrieveRequest(user_id="user-test", query="refund")

    with pytest.raises(ValidationError):
        RagRetrieveRequest(kb_ids=[], user_id="user-test", query="refund")
