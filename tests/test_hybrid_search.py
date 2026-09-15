from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.config import Settings
from app.providers.embedding.base import EmbeddingProvider
from app.providers.keyword_search.base import KeywordSearchProvider
from app.providers.vectorstores.base import VectorStore
from app.schemas.rag import RagRetrieveRequest
from app.services.hybrid_search_service import HybridSearchService
from app.services.rag_service import RagService


def _chunk(chunk_id: str, *, score: float = 0.0) -> dict:
    return {
        "document_id": "document-test",
        "chunk_id": chunk_id,
        "title": f"Title {chunk_id}",
        "content": f"Content {chunk_id}",
        "score": score,
    }


def test_hybrid_search_fuses_by_chunk_id_with_rrf() -> None:
    service = HybridSearchService(rrf_k=60)

    result = service.fuse(
        [_chunk("chunk-1", score=0.86), _chunk("chunk-2", score=0.70)],
        [
            {**_chunk("chunk-2"), "bm25_score": 12.4},
            {**_chunk("chunk-3"), "bm25_score": 9.8},
        ],
    )

    assert [chunk["chunk_id"] for chunk in result] == ["chunk-2", "chunk-1", "chunk-3"]
    assert result[0] == {
        "document_id": "document-test",
        "chunk_id": "chunk-2",
        "title": "Title chunk-2",
        "content": "Content chunk-2",
        "score": pytest.approx(1 / 62 + 1 / 61),
        "vector_score": 0.70,
        "bm25_score": 12.4,
        "vector_rank": 2,
        "bm25_rank": 1,
        "retrieval_source": "hybrid",
    }
    assert result[1]["retrieval_source"] == "vector"
    assert result[2]["retrieval_source"] == "bm25"


class FakeEmbeddingProvider(EmbeddingProvider):
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.query_calls = 0

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    async def embed_query(self, query: str) -> list[float]:
        self.query_calls += 1
        if self.fail:
            raise AssertionError("BM25 mode must not call embedding")
        return [1.0]


class FakeVectorStore(VectorStore):
    def __init__(self) -> None:
        self.calls: list[dict] = []

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
        return [_chunk("chunk-1", score=0.9), _chunk("chunk-2", score=0.8)][:top_k]

    async def delete_by_document_id(self, document_id: str) -> None:
        del document_id


class FakeKeywordSearchProvider(KeywordSearchProvider):
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[dict] = []

    async def add_chunks(self, chunks: list[dict]) -> None:
        del chunks

    async def keyword_search(self, *, query, tenant_id, kb_id, top_k=20):
        self.calls.append(
            {
                "query": query,
                "tenant_id": tenant_id,
                "kb_id": kb_id,
                "top_k": top_k,
            }
        )
        if self.fail:
            raise RuntimeError("elasticsearch unavailable")
        return [
            {**_chunk("chunk-2"), "bm25_score": 12.4},
            {**_chunk("chunk-3"), "bm25_score": 9.8},
        ][:top_k]

    async def delete_by_document_id(self, document_id: str) -> None:
        del document_id


def _rag_service(
    *,
    embedding_provider: EmbeddingProvider | None = None,
    keyword_search_provider: KeywordSearchProvider | None = None,
    retrieval_mode: str = "vector",
) -> RagService:
    settings = Settings(
        top_k=5,
        retrieval_mode=retrieval_mode,
        hybrid_rrf_k=60,
        hybrid_vector_top_k=20,
        hybrid_bm25_top_k=20,
        hybrid_top_n=20,
        rerank_enabled=False,
    )
    return RagService(
        session=SimpleNamespace(commit=AsyncMock()),
        settings=settings,
        knowledge_base_repository=SimpleNamespace(
            get_by_id=AsyncMock(return_value=SimpleNamespace(id="kb-test"))
        ),
        retrieval_log_repository=SimpleNamespace(create=AsyncMock()),
        embedding_provider=embedding_provider or FakeEmbeddingProvider(),
        vector_store=FakeVectorStore(),
        keyword_search_provider=keyword_search_provider,
    )


@pytest.mark.asyncio
async def test_rag_service_runs_vector_and_bm25_in_hybrid_mode() -> None:
    embedding_provider = FakeEmbeddingProvider()
    vector_store = FakeVectorStore()
    keyword_provider = FakeKeywordSearchProvider()
    settings = Settings(
        top_k=5,
        retrieval_mode="vector",
        hybrid_rrf_k=60,
        hybrid_vector_top_k=20,
        hybrid_bm25_top_k=20,
        hybrid_top_n=20,
    )
    service = RagService(
        session=SimpleNamespace(commit=AsyncMock()),
        settings=settings,
        knowledge_base_repository=SimpleNamespace(
            get_by_id=AsyncMock(return_value=SimpleNamespace(id="kb-test"))
        ),
        retrieval_log_repository=SimpleNamespace(create=AsyncMock()),
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        keyword_search_provider=keyword_provider,
    )

    response = await service.retrieve(
        RagRetrieveRequest(
            tenant_id="tenant-test",
            kb_id="kb-test",
            user_id="user-test",
            query="refund",
            top_k=3,
            retrieval_options={
                "mode": "hybrid",
                "vector_top_k": 2,
                "bm25_top_k": 2,
                "rrf_k": 60,
            },
        )
    )

    assert embedding_provider.query_calls == 1
    assert vector_store.calls[0]["top_k"] == 2
    assert keyword_provider.calls[0]["top_k"] == 2
    assert [chunk.chunk_id for chunk in response.retrieved_chunks] == [
        "chunk-2",
        "chunk-1",
        "chunk-3",
    ]
    assert response.retrieved_chunks[0].retrieval_source == "hybrid"
    assert response.retrieved_chunks[0].vector_rank == 2
    assert response.retrieved_chunks[0].bm25_rank == 1
    assert response.metadata["retrieval"] == {
        "mode": "hybrid",
        "fusion": "rrf",
        "rrf_k": 60,
        "vector_store": "pgvector",
        "keyword_search": "elasticsearch",
        "vector_top_k": 2,
        "bm25_top_k": 2,
        "vector_count": 2,
        "bm25_count": 2,
        "fused_count": 3,
    }


@pytest.mark.asyncio
async def test_hybrid_search_degrades_to_vector_results_when_bm25_fails() -> None:
    keyword_provider = FakeKeywordSearchProvider(fail=True)
    service = _rag_service(
        embedding_provider=FakeEmbeddingProvider(),
        keyword_search_provider=keyword_provider,
    )

    response = await service.retrieve(
        RagRetrieveRequest(
            tenant_id="tenant-test",
            kb_id="kb-test",
            user_id="user-test",
            query="refund",
            retrieval_options={"mode": "hybrid"},
        )
    )

    assert [chunk.chunk_id for chunk in response.retrieved_chunks] == ["chunk-1", "chunk-2"]
    assert all(chunk.retrieval_source == "vector" for chunk in response.retrieved_chunks)
    assert response.metadata["retrieval"]["degraded"] is True
    assert response.metadata["retrieval"]["degraded_reason"] == "bm25 search failed"


@pytest.mark.asyncio
async def test_bm25_mode_does_not_call_embedding() -> None:
    embedding_provider = FakeEmbeddingProvider(fail=True)
    keyword_provider = FakeKeywordSearchProvider()
    service = _rag_service(
        embedding_provider=embedding_provider,
        keyword_search_provider=keyword_provider,
        retrieval_mode="bm25",
    )

    response = await service.retrieve(
        RagRetrieveRequest(
            tenant_id="tenant-test",
            kb_id="kb-test",
            user_id="user-test",
            query="refund",
        )
    )

    assert embedding_provider.query_calls == 0
    assert [chunk.chunk_id for chunk in response.retrieved_chunks] == ["chunk-2", "chunk-3"]
    assert all(chunk.retrieval_source == "bm25" for chunk in response.retrieved_chunks)
