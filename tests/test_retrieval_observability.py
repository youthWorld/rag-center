from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.config import Settings
from app.providers.embedding.base import EmbeddingProvider
from app.providers.vectorstores.base import VectorStore
from app.schemas.rag import RagRetrieveRequest
from app.services.rag_service import RagService


class FakeEmbeddingProvider(EmbeddingProvider):
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    async def embed_query(self, query: str) -> list[float]:
        return [float(len(query))]


class FakeVectorStore(VectorStore):
    async def add_chunks(self, chunks: list[dict]) -> None:
        del chunks

    async def similarity_search(self, query_vector, *, tenant_id, kb_id, top_k=5):
        del query_vector, tenant_id, kb_id, top_k
        return [
            {
                "document_id": "doc-1",
                "chunk_id": "chunk-1",
                "title": "Title 1",
                "content": "secret content should not be in trace output",
                "score": 0.9,
            }
        ]

    async def delete_by_document_id(self, document_id: str) -> None:
        del document_id


class FakeTrace:
    id = "trace-test"

    def __init__(self) -> None:
        self.metadata = None
        self.spans: list[str] = []
        self.updates: list[dict] = []

    def span(self, *, name: str):
        self.spans.append(name)
        return SimpleNamespace(end=lambda **kwargs: self.updates.append({name: kwargs}))

    def update(self, **kwargs) -> None:
        self.updates.append(kwargs)


class FakeLangfuseClient:
    def __init__(self) -> None:
        self.trace_instance = FakeTrace()
        self.trace_calls: list[dict] = []
        self.flush_count = 0

    def trace(self, **kwargs):
        self.trace_calls.append(kwargs)
        return self.trace_instance

    def flush(self) -> None:
        self.flush_count += 1


class FakeRetrievalLogRepository:
    def __init__(self) -> None:
        self.create_calls: list[dict] = []

    async def create(self, **kwargs):
        self.create_calls.append(kwargs)
        return SimpleNamespace(id="log-test")


@pytest.mark.asyncio
async def test_retrieve_records_trace_spans_and_links_log_id(monkeypatch) -> None:
    client = FakeLangfuseClient()
    monkeypatch.setattr(
        "app.observability.langfuse_client.get_langfuse_client",
        lambda _settings: client,
    )
    log_repository = FakeRetrievalLogRepository()
    service = RagService(
        session=SimpleNamespace(commit=AsyncMock()),
        settings=Settings(
            top_k=1,
            retrieval_mode="vector",
            langfuse_enabled=True,
            langfuse_public_key="pk-test",
            langfuse_secret_key="sk-test",
        ),
        knowledge_base_repository=SimpleNamespace(
            get_by_id=AsyncMock(return_value=SimpleNamespace(id="kb-test"))
        ),
        retrieval_log_repository=log_repository,
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=FakeVectorStore(),
    )

    response = await service.retrieve(
        RagRetrieveRequest(kb_id="kb-test", user_id="user-test", query="question"),
        tenant_id="tenant-test",
    )

    assert response.metadata["log_id"] == "log-test"
    assert response.metadata["trace_id"] == "trace-test"
    assert client.trace_calls[0]["name"] == "rag_retrieve"
    assert client.trace_calls[0]["metadata"] == {
        "tenant_id": "tenant-test",
        "kb_id": "kb-test",
        "user_id": "user-test",
        "profile": "custom",
        "plan": "pro",
    }
    assert client.trace_instance.spans == ["query_processing", "retrieval", "rerank"]
    assert log_repository.create_calls[0]["trace_id"] == "trace-test"
    assert log_repository.create_calls[0]["profile"] == "custom"
    assert log_repository.create_calls[0]["search_query"] == "question"
    assert log_repository.create_calls[0]["effective_query"] == "question"
    assert client.flush_count == 1

    final_update = client.trace_instance.updates[-1]
    assert final_update["metadata"] == {"log_id": "log-test"}
    assert final_update["output"] == {"chunks": [{"chunk_id": "chunk-1", "score": 0.9}]}
