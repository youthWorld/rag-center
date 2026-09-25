import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.config import Settings
from app.core.error_codes import ErrorCode
from app.core.exceptions import AppError
from app.providers.embedding.base import EmbeddingProvider
from app.providers.llm.base import LLMCallMetadata, LLMJSONResponse, LLMProvider
from app.providers.vectorstores.base import VectorStore
from app.schemas.rag import EvidenceOptions, RagRetrieveRequest
from app.services.evidence_orchestration_service import (
    EvidenceOrchestrationService,
    EvidenceValidationError,
)
from app.services.rag_service import RagService
from app.tenant.plan_resolver import PlanResolver


def _candidate(
    chunk_id: str,
    *,
    kb_id: str = "kb-a",
    index_version: str = "v2",
    content: str | None = None,
    source: str = "hybrid",
) -> dict:
    return {
        "chunk_id": chunk_id,
        "document_id": f"doc-{chunk_id}",
        "kb_id": kb_id,
        "title": f"Title {chunk_id}",
        "content": content or f"Original {chunk_id}",
        "retrieval_text": f"Retrieval {chunk_id}",
        "index_version": index_version,
        "retrieval_source": source,
        "score": 1.0,
        "metadata": {"heading_path": f"Section/{chunk_id}"},
    }


def _chunk(candidate: dict, *, content: str | None = None):
    return SimpleNamespace(
        id=candidate["chunk_id"],
        tenant_id="tenant-test",
        kb_id=candidate["kb_id"],
        document_id=candidate["document_id"],
        title=candidate["title"],
        content=content or candidate["content"],
        index_version=candidate["index_version"],
        chunk_metadata={"heading_path": f"DB/{candidate['chunk_id']}"},
    )


class FakeLLMProvider(LLMProvider):
    def __init__(self, output: dict | None = None, *, fail: bool = False) -> None:
        self.output = output or {
            "aspects": [{"aspect": "refund", "evidence": [{"index": 0, "role": "core"}]}]
        }
        self.fail = fail
        self.calls: list[dict] = []

    async def chat_json(self, **kwargs) -> dict:
        return (await self.chat_json_with_metadata(**kwargs)).output

    async def chat_json_with_metadata(self, **kwargs) -> LLMJSONResponse:
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("secret original body")
        return LLMJSONResponse(
            output=self.output,
            metadata=LLMCallMetadata(
                provider="fake",
                request_model="fake-model",
                response_model="fake-model",
                latency_ms=7,
                request_id="request-1",
                finish_reason="stop",
                input_tokens=11,
                output_tokens=5,
                total_tokens=16,
            ),
        )


class FakeChunkRepository:
    def __init__(self, chunks: list[object]) -> None:
        self.chunks = chunks
        self.calls: list[dict] = []

    async def get_by_scopes(self, *, tenant_id: str, scopes: dict):
        self.calls.append({"tenant_id": tenant_id, "scopes": scopes})
        allowed = {
            (kb_id, version, chunk_id)
            for (kb_id, version), chunk_ids in scopes.items()
            for chunk_id in chunk_ids
        }
        return [
            chunk
            for chunk in self.chunks
            if (chunk.kb_id, chunk.index_version, chunk.id) in allowed
        ]


class FakeEmbeddingProvider(EmbeddingProvider):
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    async def embed_query(self, query: str) -> list[float]:
        return [float(len(query))]


class FakeVectorStore(VectorStore):
    def __init__(self, chunks: list[dict]) -> None:
        self.chunks = chunks

    async def add_chunks(self, chunks: list[dict]) -> None:
        self.chunks.extend(chunks)

    async def similarity_search(self, query_vector, *, tenant_id, kb_id, top_k=5):
        return [dict(chunk) for chunk in self.chunks[:top_k]]

    async def delete_by_document_id(self, document_id: str) -> None:
        del document_id


class FakeKnowledgeBaseRepository:
    async def get_by_id(self, *, kb_id: str, tenant_id: str):
        return SimpleNamespace(id=kb_id, name="KB")


def _service(llm: FakeLLMProvider, chunks: list[object]):
    repository = FakeChunkRepository(chunks)
    return (
        EvidenceOrchestrationService(
            SimpleNamespace(),
            llm,
            timeout_seconds=3,
            chunk_repository=repository,
        ),
        repository,
    )


def test_candidate_pool_contains_topk_context_and_outside_base_candidates() -> None:
    retrieved = [_candidate(f"top-{index}") for index in range(1, 13)]
    retrieved[0]["context"] = {
        "sources": [
            {"chunk_id": "top-1", "relation": "anchor", "content": "same"},
            {"chunk_id": "context-1", "relation": "reference", "content": "context"},
        ]
    }
    base = [*retrieved, *[_candidate(f"base-{index}") for index in range(1, 9)]]

    pool = EvidenceOrchestrationService._build_candidate_pool(
        kb_ids=["kb-a"],
        index_versions={"kb-a": "v2"},
        candidates=base,
        retrieved_chunks=retrieved,
    )

    assert len(pool) == 20
    assert [item["chunk_id"] for item in pool[:10]] == [f"top-{index}" for index in range(1, 11)]
    assert pool[10]["chunk_id"] == "context-1"
    assert pool[10]["source"] == "context:reference"
    assert [item["chunk_id"] for item in pool[11:17]] == [f"base-{index}" for index in range(1, 7)]
    assert any(item["chunk_id"] == "top-11" for item in pool)


def test_candidate_pool_preserves_outside_quota_after_cross_category_deduplication() -> None:
    retrieved = [_candidate(f"top-{index}") for index in range(1, 11)]
    retrieved[0]["context"] = {
        "sources": [
            {
                "chunk_id": "context-duplicate",
                "relation": "reference",
                "content": retrieved[0]["content"],
            },
            *[
                {
                    "chunk_id": f"context-{index}",
                    "relation": "reference",
                    "content": f"Context {index}",
                }
                for index in range(1, 5)
            ],
        ]
    }
    outside = [
        _candidate("base-duplicate", content=retrieved[1]["content"]),
        *[_candidate(f"base-{index}") for index in range(1, 7)],
    ]

    pool = EvidenceOrchestrationService._build_candidate_pool(
        kb_ids=["kb-a"],
        index_versions={"kb-a": "v2"},
        candidates=[*retrieved, *outside],
        retrieved_chunks=retrieved,
    )

    assert len(pool) == 20
    assert [item["chunk_id"] for item in pool[10:14]] == [
        f"context-{index}" for index in range(1, 5)
    ]
    assert [item["chunk_id"] for item in pool[14:]] == [f"base-{index}" for index in range(1, 7)]


@pytest.mark.asyncio
async def test_orchestrate_uses_database_content_and_allows_shared_item() -> None:
    selected = _candidate("outside", content="candidate body")
    llm = FakeLLMProvider(
        {
            "aspects": [
                {"aspect": "points", "evidence": [{"index": 1, "role": "core"}]},
                {"aspect": "cash", "evidence": [{"index": 1, "role": "supporting"}]},
            ]
        }
    )
    service, repository = _service(llm, [_chunk(selected, content="trusted database body")])

    result = await service.orchestrate(
        query="refund question",
        tenant_id="tenant-test",
        kb_ids=["kb-a"],
        index_versions={"kb-a": "v2"},
        candidates=[_candidate("top"), selected],
        retrieved_chunks=[_candidate("top")],
        options=EvidenceOptions(enabled=True, max_items=2),
    )

    assert result.pack is not None
    assert result.pack.status == "partial"
    assert len(result.pack.items) == 1
    assert result.pack.items[0].content == "trusted database body"
    assert result.pack.items[0].source == "hybrid"
    assert result.pack.items[0].retrieved_rank is None
    assert [item.evidence_id for group in result.pack.groups for item in group.evidence] == [
        "E1",
        "E1",
    ]
    assert repository.calls == [
        {
            "tenant_id": "tenant-test",
            "scopes": {("kb-a", "v2"): {"outside"}},
        }
    ]
    assert result.metadata.model_call["total_tokens"] == 16


@pytest.mark.asyncio
async def test_orchestrate_rejects_missing_or_wrong_scope_and_degrades() -> None:
    candidate = _candidate("selected")
    llm = FakeLLMProvider()
    service, _ = _service(llm, [])

    result = await service.orchestrate(
        query="question",
        tenant_id="tenant-test",
        kb_ids=["kb-a"],
        index_versions={"kb-a": "v2"},
        candidates=[candidate],
        retrieved_chunks=[candidate],
        options=EvidenceOptions(enabled=True),
    )

    assert result.pack is None
    assert result.metadata.degraded is True
    assert result.metadata.error == "selected chunks failed scope validation"


@pytest.mark.asyncio
async def test_invalid_llm_output_and_failure_do_not_leak_content(caplog) -> None:
    body = "VERY_PRIVATE_CHUNK_BODY"
    candidate = _candidate("selected", content=body)
    llm = FakeLLMProvider(fail=True)
    service, _ = _service(llm, [_chunk(candidate)])

    with caplog.at_level(logging.INFO):
        result = await service.orchestrate(
            query="VERY_PRIVATE_QUESTION",
            tenant_id="tenant-test",
            kb_ids=["kb-a"],
            index_versions={"kb-a": "v2"},
            candidates=[candidate],
            retrieved_chunks=[candidate],
            options=EvidenceOptions(enabled=True),
        )

    assert result.pack is None
    assert result.metadata.degraded is True
    assert body not in caplog.text
    assert "VERY_PRIVATE_QUESTION" not in caplog.text
    assert "secret original body" not in result.metadata.error

    with pytest.raises(EvidenceValidationError):
        EvidenceOrchestrationService._validate_output(
            {"aspects": [{"aspect": "x", "evidence": [{"index": 1, "role": "core"}]}]},
            candidate_count=1,
            max_items=8,
        )


def test_evidence_request_priority_and_plan_restriction() -> None:
    service = RagService.__new__(RagService)
    service.settings = Settings(evidence_enabled=True)
    assert (
        service._resolve_evidence_options(
            RagRetrieveRequest(kb_id="kb-a", user_id="u", query="q")
        ).enabled
        is True
    )
    assert (
        service._resolve_evidence_options(
            RagRetrieveRequest(
                kb_id="kb-a",
                user_id="u",
                query="q",
                evidence_options={"enabled": False},
            )
        ).enabled
        is False
    )


@pytest.mark.asyncio
async def test_standard_plan_rejects_enabled_evidence_but_allows_disabled() -> None:
    plan = await PlanResolver(fallback_plan="standard").resolve_for_tenant_id("tenant_demo")
    service = RagService.__new__(RagService)
    with pytest.raises(AppError) as raised:
        service._enforce_plan_features(
            plan,
            profile="custom",
            mode="vector",
            rerank_enabled=False,
            query_rewrite_enabled=False,
            evidence_enabled=True,
        )
    assert raised.value.code == ErrorCode.FEATURE_NOT_ALLOWED.code
    assert raised.value.data["feature"] == "evidence"

    service._enforce_plan_features(
        plan,
        profile="custom",
        mode="vector",
        rerank_enabled=False,
        query_rewrite_enabled=False,
        evidence_enabled=False,
    )


@pytest.mark.asyncio
async def test_disabled_evidence_does_not_call_llm() -> None:
    llm = FakeLLMProvider()
    candidate = _candidate("top")
    service, _ = _service(llm, [_chunk(candidate)])
    result = await service.orchestrate(
        query="question",
        tenant_id="tenant-test",
        kb_ids=["kb-a"],
        index_versions={"kb-a": "v2"},
        candidates=[candidate],
        retrieved_chunks=[candidate],
        options=EvidenceOptions(enabled=False),
    )
    assert result.pack is None
    assert result.metadata.enabled is False
    assert result.metadata.executed is False
    assert llm.calls == []


def _rag_service_with_evidence(*, llm_fail: bool = False):
    top = _candidate("top-1", index_version="v1")
    second = _candidate("top-2", index_version="v1")
    outside = _candidate("outside", index_version="v1")
    llm = FakeLLMProvider(
        {"aspects": [{"aspect": "outside fact", "evidence": [{"index": 2, "role": "core"}]}]},
        fail=llm_fail,
    )
    evidence, _ = _service(llm, [_chunk(outside, content="trusted outside body")])
    log_repository = SimpleNamespace(create=AsyncMock(return_value=SimpleNamespace(id="log-1")))
    rag = RagService(
        session=SimpleNamespace(commit=AsyncMock()),
        settings=Settings(top_k=2, retrieval_mode="vector", evidence_enabled=False),
        knowledge_base_repository=FakeKnowledgeBaseRepository(),
        retrieval_log_repository=log_repository,
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=FakeVectorStore([top, second, outside]),
        evidence_orchestration_service=evidence,
    )
    return rag, llm, log_repository


@pytest.mark.asyncio
async def test_rag_service_preserves_retrieved_chunks_and_adds_outside_evidence() -> None:
    service, llm, log_repository = _rag_service_with_evidence()
    base_request = dict(
        kb_id="kb-a",
        user_id="user-test",
        query="question",
        profile="custom",
        top_k=2,
        retrieval_options={"mode": "vector", "vector_top_k": 3},
    )
    disabled = await service.retrieve(
        RagRetrieveRequest(**base_request, evidence_options={"enabled": False}),
        tenant_id="tenant-test",
    )
    enabled = await service.retrieve(
        RagRetrieveRequest(**base_request, evidence_options={"enabled": True, "max_items": 2}),
        tenant_id="tenant-test",
    )

    assert [item.model_dump() for item in enabled.retrieved_chunks] == [
        item.model_dump() for item in disabled.retrieved_chunks
    ]
    assert [item.chunk_id for item in enabled.retrieved_chunks] == ["top-1", "top-2"]
    assert enabled.evidence_pack is not None
    assert [item.chunk_id for item in enabled.evidence_pack.items] == ["outside"]
    assert enabled.evidence_pack.items[0].content == "trusted outside body"
    assert len(llm.calls) == 1
    assert disabled.metadata["evidence"]["executed"] is False
    assert enabled.metadata["evidence"]["executed"] is True
    logged = log_repository.create.await_args_list[-1].kwargs["retrieval_metadata"]["evidence"]
    assert logged["evidence_count"] == 1
    assert "trusted outside body" not in str(logged)


@pytest.mark.asyncio
async def test_rag_service_returns_original_results_when_evidence_fails() -> None:
    service, _, _ = _rag_service_with_evidence(llm_fail=True)
    response = await service.retrieve(
        RagRetrieveRequest(
            kb_id="kb-a",
            user_id="user-test",
            query="question",
            profile="custom",
            top_k=2,
            retrieval_options={"mode": "vector", "vector_top_k": 3},
            evidence_options={"enabled": True},
        ),
        tenant_id="tenant-test",
    )

    assert [item.chunk_id for item in response.retrieved_chunks] == ["top-1", "top-2"]
    assert response.evidence_pack is None
    assert response.metadata["evidence"]["degraded"] is True
    assert response.metadata["application_model_call_details"]["evidence"] == 1
