from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.config import Settings
from app.providers.embedding.base import EmbeddingProvider
from app.providers.keyword_search.base import KeywordSearchProvider
from app.providers.rerank.base import RerankProvider
from app.providers.vectorstores.base import VectorStore
from app.schemas.rag import RagRetrieveRequest
from app.services.rag_service import RagService
from app.services.retrieve_once_service import RetrieveOnceService
from app.tenant.plan_resolver import PlanResolver


class FakeEmbeddingProvider(EmbeddingProvider):
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    async def embed_query(self, query: str) -> list[float]:
        return [float(len(query))]


class FakeVectorStore(VectorStore):
    def __init__(self) -> None:
        self.chunks = [
            {
                "document_id": "doc-1",
                "chunk_id": "chunk-1",
                "title": "Title 1",
                "content": "Content 1",
                "score": 0.9,
            },
            {
                "document_id": "doc-1",
                "chunk_id": "chunk-2",
                "title": "Title 2",
                "content": "Content 2",
                "score": 0.8,
            },
        ]

    async def add_chunks(self, chunks: list[dict]) -> None:
        self.chunks.extend(chunks)

    async def similarity_search(self, query_vector, *, tenant_id, kb_id, top_k=5):
        return self.chunks[:top_k]

    async def delete_by_document_id(self, document_id: str) -> None:
        del document_id


class FakeKnowledgeBaseRepository:
    async def get_by_id(self, *, kb_id: str, tenant_id: str):
        return SimpleNamespace(id=kb_id)


class EmptyKeywordSearchProvider(KeywordSearchProvider):
    async def add_chunks(self, chunks: list[dict]) -> None:
        del chunks

    async def keyword_search(self, *, query, tenant_id, kb_id, top_k=20, index_version=None):
        del query, tenant_id, kb_id, top_k, index_version
        return []

    async def delete_by_document_id(self, document_id: str, *, index_version=None) -> None:
        del document_id, index_version


class FakeRerankProvider(RerankProvider):
    def __init__(self, *, fail: bool = False, error: Exception | None = None) -> None:
        self.fail = fail
        self.error = error
        self.calls: list[dict] = []

    async def rerank(self, *, query: str, chunks: list[dict], top_n: int) -> list[dict]:
        self.calls.append({"query": query, "chunks": chunks, "top_n": top_n})
        if self.error is not None:
            raise self.error
        if self.fail:
            raise RuntimeError("invalid JSON")
        ranked = list(reversed(chunks))
        return [
            {**chunk, "rerank_score": 0.95 if index == 0 else 0.2}
            for index, chunk in enumerate(ranked)
        ][:top_n]


def _service(
    rerank_provider: RerankProvider,
    *,
    enabled: bool = True,
    max_candidates: int = 20,
):
    settings = Settings(
        top_k=2,
        rerank_enabled=enabled,
        rerank_top_n=1,
        llm_provider="openai_compatible",
        llm_model="test-model",
    )
    service = RagService(
        session=SimpleNamespace(commit=AsyncMock()),
        settings=settings,
        knowledge_base_repository=FakeKnowledgeBaseRepository(),
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=FakeVectorStore(),
        rerank_provider=rerank_provider,
    )
    return service


@pytest.mark.asyncio
async def test_rag_service_returns_reranked_chunks_and_metadata() -> None:
    rerank_provider = FakeRerankProvider()
    service = _service(rerank_provider)

    response = await service.retrieve(
        RagRetrieveRequest(
            kb_id="kb-test",
            user_id="user-test",
            query="question",
            top_k=2,
            rerank_options={"enabled": True, "top_n": 1},
        ),
        tenant_id="tenant-test",
    )

    assert [chunk.chunk_id for chunk in response.retrieved_chunks] == ["chunk-2"]
    assert response.retrieved_chunks[0].score == 0.8
    assert response.retrieved_chunks[0].rerank_score == 0.95
    assert response.metadata["rerank"]["provider"] == "qwen3.7"
    assert response.metadata["rerank"]["model"] == "qwen3.7-text-rerank"
    assert response.metadata["rerank"]["candidate_count"] == 2
    assert response.metadata["rerank"]["returned_count"] == 1
    assert response.metadata["rerank"]["degraded"] is False


@pytest.mark.asyncio
async def test_rag_service_degrades_to_vector_order_when_rerank_fails() -> None:
    service = _service(FakeRerankProvider(fail=True))

    response = await service.retrieve(
        RagRetrieveRequest(
            kb_id="kb-test",
            user_id="user-test",
            query="question",
            rerank_options={"enabled": True, "top_n": 1},
        ),
        tenant_id="tenant-test",
    )

    assert [chunk.chunk_id for chunk in response.retrieved_chunks] == ["chunk-1"]
    assert all(chunk.rerank_score is None for chunk in response.retrieved_chunks)
    assert response.metadata["rerank"]["enabled"] is True
    assert response.metadata["rerank"]["degraded"] is True
    assert response.metadata["rerank"]["error"] == "rerank failed (RuntimeError)"


@pytest.mark.asyncio
async def test_request_can_disable_globally_enabled_rerank() -> None:
    rerank_provider = FakeRerankProvider()
    service = _service(rerank_provider, enabled=True)

    response = await service.retrieve(
        RagRetrieveRequest(
            kb_id="kb-test",
            user_id="user-test",
            query="question",
            rerank_options={"enabled": False},
        ),
        tenant_id="tenant-test",
    )

    assert not rerank_provider.calls
    assert [chunk.chunk_id for chunk in response.retrieved_chunks] == ["chunk-1", "chunk-2"]
    assert response.metadata["rerank"]["enabled"] is False
    assert response.metadata["rerank"]["candidate_count"] == 0


@pytest.mark.asyncio
async def test_rag_service_custom_uses_explicit_candidate_count() -> None:
    rerank_provider = FakeRerankProvider()
    service = _service(rerank_provider, max_candidates=1)

    response = await service.retrieve(
        RagRetrieveRequest(
            kb_id="kb-test",
            user_id="user-test",
            query="question",
            top_k=2,
            rerank_options={"enabled": True, "top_n": 1},
        ),
        tenant_id="tenant-test",
    )

    assert len(rerank_provider.calls) == 1
    assert [chunk["chunk_id"] for chunk in rerank_provider.calls[0]["chunks"]] == [
        "chunk-1",
        "chunk-2",
    ]
    assert response.metadata["rerank"]["candidate_count"] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [RuntimeError("HTTP 429"), TimeoutError()])
async def test_research_retrieve_once_degrades_rerank_without_retry(failure: Exception) -> None:
    rerank = FakeRerankProvider(error=failure)
    rag = _service(rerank)
    rag.keyword_search_provider = EmptyKeywordSearchProvider()
    result = await RetrieveOnceService(rag).execute(
        tenant_id="tenant-test", user_id="user-test", kb_ids=["kb-test"],
        index_versions={"kb-test": "v1"}, query_id="Q1", search_query="question",
        aspect_ids=["A1"], round=1, plan=PlanResolver().resolve("pro"),
    )
    assert result.error is None
    assert result.degraded is True
    assert result.metadata["rerank"]["degraded"] is True
    assert [chunk.chunk_id for chunk in result.retrieved_chunks][:2] == [
        "chunk-1", "chunk-2"
    ]
    assert len(rerank.calls) == 1
