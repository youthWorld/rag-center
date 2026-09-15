from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.config import Settings
from app.providers.embedding.base import EmbeddingProvider
from app.providers.rerank.base import RerankProvider
from app.providers.vectorstores.base import VectorStore
from app.schemas.rag import RagRetrieveRequest
from app.services.rag_service import RagService


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


class FakeRetrievalLogRepository:
    def __init__(self) -> None:
        self.create = AsyncMock()


class FakeRerankProvider(RerankProvider):
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[dict] = []

    async def rerank(self, *, query: str, chunks: list[dict], top_n: int) -> list[dict]:
        self.calls.append({"query": query, "chunks": chunks, "top_n": top_n})
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
        rerank_provider="llm",
        rerank_top_n=1,
        rerank_max_candidates=max_candidates,
        llm_provider="openai_compatible",
        llm_model="test-model",
    )
    log_repository = FakeRetrievalLogRepository()
    service = RagService(
        session=SimpleNamespace(commit=AsyncMock()),
        settings=settings,
        knowledge_base_repository=FakeKnowledgeBaseRepository(),
        retrieval_log_repository=log_repository,
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=FakeVectorStore(),
        rerank_provider=rerank_provider,
    )
    return service, log_repository


@pytest.mark.asyncio
async def test_rag_service_returns_reranked_chunks_and_metadata() -> None:
    rerank_provider = FakeRerankProvider()
    service, log_repository = _service(rerank_provider)

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
    assert response.metadata["rerank"] == {
        "enabled": True,
        "provider": "llm",
        "llm_provider": "openai_compatible",
        "model": "test-model",
        "top_n": 1,
        "candidate_count": 2,
    }
    assert log_repository.create.await_args.kwargs["retrieved_chunks"][0]["rerank_score"] == 0.95


@pytest.mark.asyncio
async def test_rag_service_degrades_to_vector_order_when_rerank_fails() -> None:
    service, _ = _service(FakeRerankProvider(fail=True))

    response = await service.retrieve(
        RagRetrieveRequest(
            kb_id="kb-test",
            user_id="user-test",
            query="question",
            rerank_options={"enabled": True, "top_n": 1},
        ),
        tenant_id="tenant-test",
    )

    assert [chunk.chunk_id for chunk in response.retrieved_chunks] == ["chunk-1", "chunk-2"]
    assert all(chunk.rerank_score is None for chunk in response.retrieved_chunks)
    assert response.metadata["rerank"]["enabled"] is True
    assert response.metadata["rerank"]["degraded"] is True
    assert response.metadata["rerank"]["error"] == "invalid JSON"


@pytest.mark.asyncio
async def test_request_can_disable_globally_enabled_rerank() -> None:
    rerank_provider = FakeRerankProvider()
    service, _ = _service(rerank_provider, enabled=True)

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
async def test_rag_service_limits_candidates_before_provider() -> None:
    rerank_provider = FakeRerankProvider()
    service, _ = _service(rerank_provider, max_candidates=1)

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
    assert [chunk["chunk_id"] for chunk in rerank_provider.calls[0]["chunks"]] == ["chunk-1"]
    assert response.metadata["rerank"]["candidate_count"] == 1
