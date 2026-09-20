from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.config import Settings
from app.core.error_codes import ErrorCode
from app.core.exceptions import AppError
from app.providers.embedding.base import EmbeddingProvider
from app.providers.keyword_search.base import KeywordSearchProvider
from app.providers.vectorstores.base import VectorStore
from app.schemas.rag import RagRetrieveRequest
from app.services.rag_service import RagService


def _chunk(kb_id: str, chunk_id: str, *, score: float = 0.9) -> dict:
    return {
        "document_id": f"document-{kb_id}",
        "chunk_id": chunk_id,
        "title": f"Title {chunk_id}",
        "content": f"Content {chunk_id}",
        "score": score,
    }


class FakeEmbeddingProvider(EmbeddingProvider):
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[str] = []

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    async def embed_query(self, query: str) -> list[float]:
        self.calls.append(query)
        if self.fail:
            raise RuntimeError("embedding unavailable")
        return [1.0]


class FakeVectorStore(VectorStore):
    def __init__(
        self,
        results: dict[str, list[dict]] | None = None,
        *,
        fail_kb_ids: set[str] | None = None,
    ) -> None:
        self.results = results or {}
        self.fail_kb_ids = fail_kb_ids or set()
        self.calls: list[str] = []

    async def add_chunks(self, chunks: list[dict]) -> None:
        del chunks

    async def similarity_search(
        self,
        query_vector,
        *,
        tenant_id,
        kb_id,
        top_k=5,
    ) -> list[dict]:
        del query_vector, tenant_id
        self.calls.append(kb_id)
        if kb_id in self.fail_kb_ids:
            raise RuntimeError(f"vector unavailable for {kb_id}")
        return list(self.results.get(kb_id, []))[:top_k]

    async def delete_by_document_id(self, document_id: str) -> None:
        del document_id


class FakeKeywordSearchProvider(KeywordSearchProvider):
    def __init__(
        self,
        results: dict[str, list[dict]] | None = None,
        *,
        fail_kb_ids: set[str] | None = None,
        fail_all: bool = False,
    ) -> None:
        self.results = results or {}
        self.fail_kb_ids = fail_kb_ids or set()
        self.fail_all = fail_all
        self.calls: list[str] = []

    async def add_chunks(self, chunks: list[dict]) -> None:
        del chunks

    async def keyword_search(
        self,
        *,
        query,
        tenant_id,
        kb_id,
        top_k=20,
    ) -> list[dict]:
        del query, tenant_id
        self.calls.append(kb_id)
        if self.fail_all or kb_id in self.fail_kb_ids:
            raise RuntimeError(f"bm25 unavailable for {kb_id}")
        return list(self.results.get(kb_id, []))[:top_k]

    async def delete_by_document_id(self, document_id: str) -> None:
        del document_id


class FakeKnowledgeBaseRepository:
    def __init__(self, kb_ids: list[str], *, indexed_counts: dict[str, int] | None = None) -> None:
        self.knowledge_bases = {
            kb_id: SimpleNamespace(id=kb_id, name=f"Name {kb_id}") for kb_id in kb_ids
        }
        self.indexed_counts = indexed_counts or {}

    async def get_by_id(self, *, kb_id: str, tenant_id: str):
        del tenant_id
        return self.knowledge_bases.get(kb_id)

    async def get_by_ids(self, *, kb_ids: list[str], tenant_id: str):
        del tenant_id
        return [self.knowledge_bases[kb_id] for kb_id in kb_ids if kb_id in self.knowledge_bases]

    async def count_indexed_chunks(self, *, kb_id: str, tenant_id: str) -> int:
        del tenant_id
        return self.indexed_counts.get(kb_id, 0)


def _service(
    *,
    kb_ids: list[str] | None = None,
    vector_results: dict[str, list[dict]] | None = None,
    bm25_results: dict[str, list[dict]] | None = None,
    vector_fail_kb_ids: set[str] | None = None,
    bm25_fail_kb_ids: set[str] | None = None,
    bm25_fail_all: bool = False,
    embedding_fail: bool = False,
    indexed_counts: dict[str, int] | None = None,
) -> RagService:
    selected_kb_ids = kb_ids or ["kb-test"]
    return RagService(
        session=SimpleNamespace(commit=AsyncMock()),
        settings=Settings(
            top_k=5,
            retrieval_mode="vector",
            hybrid_rrf_k=60,
            hybrid_vector_top_k=20,
            hybrid_bm25_top_k=20,
            hybrid_top_n=20,
            query_max_length=2000,
            rerank_enabled=False,
        ),
        knowledge_base_repository=FakeKnowledgeBaseRepository(
            selected_kb_ids,
            indexed_counts=indexed_counts,
        ),
        retrieval_log_repository=SimpleNamespace(
            create=AsyncMock(return_value=SimpleNamespace(id="log-test"))
        ),
        embedding_provider=FakeEmbeddingProvider(fail=embedding_fail),
        vector_store=FakeVectorStore(
            vector_results,
            fail_kb_ids=vector_fail_kb_ids,
        ),
        keyword_search_provider=FakeKeywordSearchProvider(
            bm25_results,
            fail_kb_ids=bm25_fail_kb_ids,
            fail_all=bm25_fail_all,
        ),
    )


def _request(*, kb_id: str = "kb-test", query: str = "refund", multi: bool = False):
    if multi:
        return RagRetrieveRequest(
            kb_ids=["kb-a", "kb-b"],
            user_id="user-test",
            query=query,
            profile="custom",
            retrieval_options={"mode": "vector"},
        )
    return RagRetrieveRequest(
        kb_id=kb_id,
        user_id="user-test",
        query=query,
        profile="custom",
        retrieval_options={"mode": "hybrid"},
    )


@pytest.mark.asyncio
async def test_hybrid_degrades_to_bm25_when_vector_fails() -> None:
    service = _service(
        vector_results={"kb-test": [_chunk("kb-test", "vector-1")]},
        bm25_results={"kb-test": [_chunk("kb-test", "bm25-1")]},
        vector_fail_kb_ids={"kb-test"},
    )

    response = await service.retrieve(_request(), tenant_id="tenant-test")

    assert [chunk.chunk_id for chunk in response.retrieved_chunks] == ["bm25-1"]
    assert all(chunk.retrieval_source == "bm25" for chunk in response.retrieved_chunks)
    assert response.metadata["retrieval"]["degraded"] is True
    assert response.metadata["retrieval"]["degraded_reason"] == "vector search failed"


@pytest.mark.asyncio
async def test_hybrid_degrades_to_vector_when_bm25_fails() -> None:
    service = _service(
        vector_results={"kb-test": [_chunk("kb-test", "vector-1")]},
        bm25_results={"kb-test": [_chunk("kb-test", "bm25-1")]},
        bm25_fail_all=True,
    )

    response = await service.retrieve(_request(), tenant_id="tenant-test")

    assert [chunk.chunk_id for chunk in response.retrieved_chunks] == ["vector-1"]
    assert all(chunk.retrieval_source == "vector" for chunk in response.retrieved_chunks)
    assert response.metadata["retrieval"]["degraded"] is True
    assert response.metadata["retrieval"]["degraded_reason"] == "bm25 search failed"


@pytest.mark.asyncio
async def test_embedding_failure_degrades_hybrid_to_bm25() -> None:
    service = _service(
        bm25_results={"kb-test": [_chunk("kb-test", "bm25-1")]},
        embedding_fail=True,
    )

    response = await service.retrieve(_request(), tenant_id="tenant-test")

    assert [chunk.chunk_id for chunk in response.retrieved_chunks] == ["bm25-1"]
    assert response.metadata["retrieval"]["degraded_reason"] == "vector search failed"


@pytest.mark.asyncio
async def test_hybrid_rejects_when_vector_and_bm25_both_fail() -> None:
    service = _service(
        vector_fail_kb_ids={"kb-test"},
        bm25_fail_all=True,
    )

    with pytest.raises(AppError) as raised:
        await service.retrieve(_request(), tenant_id="tenant-test")

    assert raised.value.code == ErrorCode.RETRIEVAL_FAILED.code


@pytest.mark.asyncio
async def test_multi_kb_partial_failure_keeps_successful_chunks() -> None:
    service = _service(
        kb_ids=["kb-a", "kb-b"],
        vector_results={"kb-b": [_chunk("kb-b", "chunk-b")]},
        vector_fail_kb_ids={"kb-a"},
    )

    response = await service.retrieve(
        _request(multi=True),
        tenant_id="tenant-test",
    )

    assert [chunk.chunk_id for chunk in response.retrieved_chunks] == ["chunk-b"]
    retrieval = response.metadata["retrieval"]
    assert retrieval["partial_kb_success"] is True
    assert retrieval["failed_kb_ids"] == ["kb-a"]
    assert retrieval["degraded"] is True
    assert retrieval["per_kb_metadata"]["kb-a"]["error_type"] == "AppError"


@pytest.mark.asyncio
async def test_multi_kb_rejects_when_all_knowledge_bases_fail() -> None:
    service = _service(
        kb_ids=["kb-a", "kb-b"],
        vector_fail_kb_ids={"kb-a", "kb-b"},
    )

    with pytest.raises(AppError) as raised:
        await service.retrieve(
            _request(multi=True),
            tenant_id="tenant-test",
        )

    assert raised.value.code == ErrorCode.RETRIEVAL_FAILED.code


@pytest.mark.asyncio
async def test_empty_query_raises_param_error() -> None:
    service = _service()

    with pytest.raises(AppError) as raised:
        await service.retrieve(
            _request(query="   "),
            tenant_id="tenant-test",
        )

    assert raised.value.code == ErrorCode.PARAM_ERROR.code


@pytest.mark.asyncio
async def test_overlong_query_raises_param_error() -> None:
    service = _service()

    with pytest.raises(AppError) as raised:
        await service.retrieve(
            _request(query="q" * 2001),
            tenant_id="tenant-test",
        )

    assert raised.value.code == ErrorCode.PARAM_ERROR.code


@pytest.mark.asyncio
async def test_empty_result_reports_no_chunks_matched() -> None:
    service = _service(
        vector_results={"kb-test": []},
        bm25_results={"kb-test": []},
        indexed_counts={"kb-test": 3},
    )

    response = await service.retrieve(_request(), tenant_id="tenant-test")

    assert response.retrieved_chunks == []
    assert response.metadata["retrieval"]["empty_reason"] == "no_chunks_matched"


@pytest.mark.asyncio
async def test_empty_result_reports_no_indexed_chunks() -> None:
    service = _service(
        vector_results={"kb-test": []},
        bm25_results={"kb-test": []},
        indexed_counts={"kb-test": 0},
    )

    response = await service.retrieve(_request(), tenant_id="tenant-test")

    assert response.retrieved_chunks == []
    assert response.metadata["retrieval"]["empty_reason"] == "no_indexed_chunks"
