from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import app.api.dependencies as api_dependencies
from app.core.config import Settings
from app.core.error_codes import ErrorCode
from app.core.exceptions import AppError
from app.providers.embedding.base import EmbeddingProvider
from app.providers.keyword_search.base import KeywordSearchProvider
from app.providers.rerank.base import RerankProvider
from app.providers.vectorstores.base import VectorStore
from app.schemas.rag import RagRetrieveRequest
from app.services.quota_service import QuotaService
from app.services.rag_service import RagService
from app.services.rate_limit_service import RateLimitService
from app.tenant.plan_resolver import PlanResolver, resolve_plan


class FakeTenantRepository:
    def __init__(self, plan: str) -> None:
        self.tenant = SimpleNamespace(id="tenant-test", plan=plan)

    async def get_by_id(self, tenant_id: str):
        assert tenant_id == self.tenant.id
        return self.tenant


class FakeEmbeddingProvider(EmbeddingProvider):
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    async def embed_query(self, query: str) -> list[float]:
        return [float(len(query))]


class FakeVectorStore(VectorStore):
    async def add_chunks(self, chunks: list[dict]) -> None:
        del chunks

    async def delete_by_document_id(self, document_id: str) -> None:
        del document_id

    async def similarity_search(self, query_vector, *, tenant_id, kb_id, top_k=5):
        del query_vector, tenant_id, kb_id
        return [
            {
                "document_id": "doc-1",
                "chunk_id": "chunk-1",
                "title": "Title",
                "content": "Content",
                "score": 0.9,
            }
        ][:top_k]


class FakeKeywordSearchProvider(KeywordSearchProvider):
    async def add_chunks(self, chunks: list[dict]) -> None:
        del chunks

    async def delete_by_document_id(self, document_id: str) -> None:
        del document_id

    async def keyword_search(self, *, query: str, tenant_id: str, kb_id: str, top_k: int):
        del query, tenant_id, kb_id
        return [
            {
                "document_id": "doc-1",
                "chunk_id": "chunk-1",
                "title": "Title",
                "content": "Content",
                "score": 0.8,
            }
        ][:top_k]


class FakeRerankProvider(RerankProvider):
    async def rerank(self, *, query: str, chunks: list[dict], top_n: int) -> list[dict]:
        del query
        return [{**chunk, "rerank_score": 0.99} for chunk in chunks[:top_n]]


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, int] = {}
        self.expirations: dict[str, int] = {}

    async def incr(self, key: str) -> int:
        self.values[key] = self.values.get(key, 0) + 1
        return self.values[key]

    async def expire(self, key: str, seconds: int) -> bool:
        self.expirations[key] = seconds
        return True

    async def get(self, key: str):
        return self.values.get(key)


def _rag_service(plan: str, *, keyword: bool = False) -> RagService:
    return RagService(
        session=SimpleNamespace(commit=AsyncMock()),
        settings=Settings(retrieval_mode="vector", top_k=5),
        knowledge_base_repository=SimpleNamespace(
            get_by_id=AsyncMock(return_value=SimpleNamespace(id="kb-test"))
        ),
        retrieval_log_repository=SimpleNamespace(create=AsyncMock()),
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=FakeVectorStore(),
        keyword_search_provider=FakeKeywordSearchProvider() if keyword else None,
        rerank_provider=FakeRerankProvider(),
        plan_resolver=PlanResolver(FakeTenantRepository(plan)),
    )


@pytest.mark.asyncio
async def test_free_speed_profile_succeeds_and_exposes_policy() -> None:
    service = _rag_service("free")

    response = await service.retrieve(
        RagRetrieveRequest(
            kb_id="kb-test",
            user_id="user-test",
            query="question",
            profile="speed",
        ),
        tenant_id="tenant-test",
    )

    assert response.metadata["tenant_policy"] == {
        "plan": "free",
        "retrieve_profile": "speed",
        "effective_mode": "vector",
        "effective_rerank": False,
        "effective_query_rewrite": False,
    }
    assert response.metadata["top_k"] == 3


@pytest.mark.asyncio
async def test_free_balanced_profile_is_rejected_before_search() -> None:
    service = _rag_service("free")

    with pytest.raises(AppError) as raised:
        await service.retrieve(
            RagRetrieveRequest(
                kb_id="kb-test",
                user_id="user-test",
                query="question",
                profile="balanced",
            ),
            tenant_id="tenant-test",
        )

    assert raised.value.code == ErrorCode.FEATURE_NOT_ALLOWED.code


@pytest.mark.asyncio
async def test_standard_custom_rerank_is_rejected() -> None:
    service = _rag_service("standard")

    with pytest.raises(AppError) as raised:
        await service.retrieve(
            RagRetrieveRequest(
                kb_id="kb-test",
                user_id="user-test",
                query="question",
                profile="custom",
                rerank_options={"enabled": True},
            ),
            tenant_id="tenant-test",
        )

    assert raised.value.code == ErrorCode.FEATURE_NOT_ALLOWED.code


@pytest.mark.asyncio
async def test_pro_quality_profile_enables_hybrid_rerank_and_rewrite() -> None:
    service = _rag_service("pro", keyword=True)

    response = await service.retrieve(
        RagRetrieveRequest(
            kb_id="kb-test",
            user_id="user-test",
            query="question",
            profile="quality",
        ),
        tenant_id="tenant-test",
    )

    assert response.metadata["tenant_policy"]["effective_mode"] == "hybrid"
    assert response.metadata["tenant_policy"]["effective_rerank"] is True
    assert response.metadata["tenant_policy"]["effective_query_rewrite"] is True


@pytest.mark.asyncio
async def test_rate_limit_service_uses_expected_keys_and_error_codes() -> None:
    redis = FakeRedis()

    def clock() -> datetime:
        return datetime(2026, 9, 16, 12, 0, tzinfo=UTC)

    service = RateLimitService(redis, clock=clock)
    free = resolve_plan("free")

    await service.check_retrieve("tenant-test", free)
    await service.check_retrieve("tenant-test", free)
    await service.check_retrieve("tenant-test", free)
    with pytest.raises(AppError) as raised:
        await service.check_retrieve("tenant-test", free)

    assert raised.value.code == ErrorCode.API_RATE_LIMIT.code == 20005
    assert redis.expirations["rag:ratelimit:retrieve:tenant-test:1789560000"] == 2

    await service.record_retrieve_success("tenant-test")
    assert await service.get_retrieve_daily_count("tenant-test") == 1
    assert redis.expirations["rag:quota:retrieve:daily:tenant-test:20260916"] == 25 * 60 * 60


@pytest.mark.asyncio
async def test_rate_limit_dependency_closes_redis_client(monkeypatch) -> None:
    created: list[object] = []

    class FakeRateLimitService:
        def __init__(self, *, redis_url: str) -> None:
            self.redis_url = redis_url
            self.closed = False
            created.append(self)

        async def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(api_dependencies, "RateLimitService", FakeRateLimitService)
    dependency = api_dependencies.get_rate_limit_service(
        Settings(celery_broker_url="redis://test/0")
    )

    service = await anext(dependency)
    assert service.redis_url == "redis://test/0"
    assert not service.closed

    await dependency.aclose()

    assert created == [service]
    assert service.closed


@pytest.mark.asyncio
async def test_quota_service_rejects_free_tenant_limits() -> None:
    knowledge_base_repository = SimpleNamespace(
        count_by_tenant=AsyncMock(return_value=1),
        count_documents=AsyncMock(return_value=30),
    )
    document_repository = SimpleNamespace(
        count_by_tenant_and_status=AsyncMock(return_value=1),
    )
    service = QuotaService(knowledge_base_repository, document_repository)
    free = resolve_plan("free")

    with pytest.raises(AppError) as kb_error:
        await service.check_create_knowledge_base(tenant_id="tenant-test", plan=free)
    assert kb_error.value.code == ErrorCode.QUOTA_EXCEEDED.code

    with pytest.raises(AppError) as document_error:
        await service.check_upload_document(
            tenant_id="tenant-test",
            kb_id="kb-test",
            plan=free,
        )
    assert document_error.value.code == ErrorCode.QUOTA_EXCEEDED.code
