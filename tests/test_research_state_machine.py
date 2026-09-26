from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from app.core.config import Settings
from app.core.error_codes import ErrorCode
from app.core.exceptions import AppError, LLMServiceError
from app.providers.llm.base import LLMCallMetadata, LLMJSONResponse, LLMProvider
from app.providers.llm.role_router import LLMRole
from app.schemas.rag import RetrievedChunk
from app.schemas.research import ResearchRequest
from app.services.research_service import ResearchService
from app.services.retrieve_once_service import RetrieveOnceResult
from app.tenant.plan_resolver import PlanResolver


class QueueLLMProvider(LLMProvider):
    def __init__(self, outputs: list[dict[str, Any] | BaseException]) -> None:
        self.outputs = list(outputs)
        self.calls: list[dict[str, Any]] = []

    async def chat_json(self, **kwargs: Any) -> dict[str, Any]:
        return (await self.chat_json_with_metadata(**kwargs)).output

    async def chat_json_with_metadata(self, **kwargs: Any) -> LLMJSONResponse:
        self.calls.append(kwargs)
        output = self.outputs.pop(0)
        if isinstance(output, BaseException):
            raise output
        return LLMJSONResponse(
            output=output,
            metadata=LLMCallMetadata(
                request_model="fake-model",
                input_tokens=10,
                output_tokens=5,
            ),
        )


class FakeRoleRouter:
    def __init__(self, fast: QueueLLMProvider, strong: QueueLLMProvider) -> None:
        self.fast = fast
        self.strong = strong

    def model_for(self, role: LLMRole) -> str:
        return "fast-model" if role == LLMRole.FAST else "strong-model"

    def provider_for(self, role: LLMRole) -> QueueLLMProvider:
        return self.fast if role == LLMRole.FAST else self.strong


class FakeKnowledgeBaseRepository:
    async def get_by_ids(self, *, tenant_id: str, kb_ids: list[str]) -> list[Any]:
        del tenant_id
        return [
            SimpleNamespace(
                id=kb_id,
                name=f"KB {kb_id}",
                description="rules",
                active_index_version="v2",
            )
            for kb_id in kb_ids
        ]


class FakeChunkRepository:
    def __init__(self, chunks: list[Any]) -> None:
        self.chunks = {
            (chunk.kb_id, chunk.index_version, chunk.id): chunk for chunk in chunks
        }

    async def get_by_scopes(
        self, *, tenant_id: str, scopes: dict[tuple[str, str], set[str]]
    ) -> list[Any]:
        del tenant_id
        return [
            chunk
            for key, chunk in self.chunks.items()
            if key[2] in scopes.get((key[0], key[1]), set())
        ]


class FakeRateLimitService:
    def __init__(self) -> None:
        self.checks = 0
        self.records = 0

    async def check_retrieve(self, tenant_id: str, plan: Any) -> None:
        del tenant_id, plan
        self.checks += 1

    async def record_retrieve_success(self, tenant_id: str) -> None:
        del tenant_id
        self.records += 1


def make_chunk(chunk_id: str, content: str) -> Any:
    return SimpleNamespace(
        id=chunk_id,
        tenant_id="tenant-a",
        kb_id="kb-a",
        document_id=f"doc-{chunk_id}",
        title=f"Title {chunk_id}",
        content=content,
        index_version="v2",
        chunk_metadata={"heading_path": f"Section {chunk_id}"},
    )


def retrieved(chunk: Any) -> RetrievedChunk:
    return RetrievedChunk(
        document_id=chunk.document_id,
        chunk_id=chunk.id,
        kb_id=chunk.kb_id,
        title=chunk.title,
        content=chunk.content,
        score=1.0,
        retrieval_source="hybrid",
        index_version=chunk.index_version,
        metadata=dict(chunk.chunk_metadata),
    )


def build_service(
    *,
    fast_outputs: list[dict[str, Any] | BaseException],
    strong_outputs: list[dict[str, Any] | BaseException],
    chunks: list[Any],
    executor: Any,
) -> tuple[ResearchService, QueueLLMProvider, QueueLLMProvider, FakeRateLimitService]:
    fast = QueueLLMProvider(fast_outputs)
    strong = QueueLLMProvider(strong_outputs)
    rate_limit = FakeRateLimitService()
    service = ResearchService(
        settings=Settings(
            llm_api_key="test",
            llm_model="fast-model",
            llm_strong_model="strong-model",
            langfuse_enabled=False,
        ),
        knowledge_base_repository=FakeKnowledgeBaseRepository(),
        chunk_repository=FakeChunkRepository(chunks),
        plan_resolver=PlanResolver(),
        rate_limit_service=rate_limit,
        retrieve_executor=executor,
        role_router=FakeRoleRouter(fast, strong),
    )
    return service, fast, strong, rate_limit


def planner_output() -> dict[str, Any]:
    return {
        "aspects": ["积分有效期"],
        "queries": [{"query": "退款积分有效期", "aspect_indexes": [0]}],
    }


def decider_output(*, role: str = "core", next_query: bool = False) -> dict[str, Any]:
    return {
        "aspects": [
            {
                "aspect_index": 0,
                "evidence": [{"candidate_index": 0, "role": role}],
            }
        ],
        "decision": "supplement" if next_query else "stop",
        "next_queries": (
            [{"query": "退回积分的有效期规则", "aspect_indexes": [0]}]
            if next_query
            else []
        ),
    }


@pytest.mark.asyncio
async def test_research_completes_in_one_round_with_traceable_citation() -> None:
    chunk = make_chunk("chunk-1", "退款积分按积分有效期规则处理。")

    async def executor(**kwargs: Any) -> RetrieveOnceResult:
        assert kwargs["index_versions"] == {"kb-a": "v2"}
        return RetrieveOnceResult(
            query_id=kwargs["query_id"],
            query=kwargs["search_query"],
            aspect_ids=kwargs["aspect_ids"],
            round=kwargs["round"],
            candidate_snapshot=[],
            retrieved_chunks=[retrieved(chunk)],
            metadata={"latency_ms": 2},
            latency_ms=2,
        )

    service, fast, strong, rate_limit = build_service(
        fast_outputs=[planner_output(), decider_output()],
        strong_outputs=[
            {
                "aspects": [
                    {
                        "aspect_index": 0,
                        "evidence": [{"candidate_index": 0, "role": "core"}],
                    }
                ],
                "answer_blocks": [
                    {
                        "text": "退款积分遵循积分有效期规则。",
                        "evidence_indexes": [0],
                        "kind": "supported",
                    }
                ],
            }
        ],
        chunks=[chunk],
        executor=executor,
    )

    result = await service.research(
        ResearchRequest(kb_id="kb-a", user_id="user-a", query="退款积分有效期？"),
        tenant_id="tenant-a",
        tenant=SimpleNamespace(plan="pro"),
    )

    assert result.answer == "退款积分遵循积分有效期规则。[E1]"
    assert result.evidence_pack.status == "complete"
    assert result.evidence_pack.items[0].chunk_id == "chunk-1"
    assert result.evidence_pack.items[0].rounds == [1]
    assert result.evidence_pack.items[0].query_ids == ["Q1"]
    assert result.metadata.stop_reason == "evidence_complete"
    assert result.metadata.first_round_missing_aspect_ids == []
    assert result.metadata.round_count == 1
    assert result.metadata.llm_call_count == 3
    assert result.rounds[0].tasks[0].new_chunk_count == 1
    assert result.rounds[0].tasks[0].chunk_count == 1
    assert "chunk_ids" not in result.rounds[0].tasks[0].model_dump()
    assert "chunks" not in result.model_dump(mode="json")
    assert rate_limit.checks == 1
    assert rate_limit.records == 1
    assert all(call["log_payload"] is False for call in [*fast.calls, *strong.calls])
    assert all(call["enable_thinking"] is False for call in [*fast.calls, *strong.calls])


@pytest.mark.asyncio
async def test_research_executes_one_supplement_round_and_merges_provenance() -> None:
    clue = make_chunk("chunk-1", "具体有效期遵循积分有效期规则。")
    rule = make_chunk("chunk-2", "退回积分有效期为原积分剩余有效期。")

    async def executor(**kwargs: Any) -> RetrieveOnceResult:
        chunk = clue if kwargs["round"] == 1 else rule
        return RetrieveOnceResult(
            query_id=kwargs["query_id"],
            query=kwargs["search_query"],
            aspect_ids=kwargs["aspect_ids"],
            round=kwargs["round"],
            candidate_snapshot=[],
            retrieved_chunks=[retrieved(chunk)],
            metadata={"latency_ms": 2},
            latency_ms=2,
        )

    service, _, _, _ = build_service(
        fast_outputs=[planner_output(), decider_output(role="supporting", next_query=True)],
        strong_outputs=[
            {
                "aspects": [
                    {
                        "aspect_index": 0,
                        "evidence": [
                            {"candidate_index": 0, "role": "supporting"},
                            {"candidate_index": 1, "role": "core"},
                        ],
                    }
                ],
                "answer_blocks": [
                    {
                        "text": "退回积分按原积分剩余有效期计算。",
                        "evidence_indexes": [1],
                        "kind": "supported",
                    }
                ],
            }
        ],
        chunks=[clue, rule],
        executor=executor,
    )

    result = await service.research(
        ResearchRequest(kb_ids=["kb-a"], user_id="user-a", query="退款积分有效期？"),
        tenant_id="tenant-a",
        tenant=SimpleNamespace(plan="pro"),
    )

    assert result.metadata.round_count == 2
    assert result.metadata.retrieval_task_count == 2
    assert result.metadata.stop_reason == "evidence_complete"
    assert result.metadata.first_round_missing_aspect_ids == ["A1"]
    assert result.rounds[1].new_chunk_count == 1
    assert result.answer == "退回积分按原积分剩余有效期计算。[E2]"
    assert [item.rounds for item in result.evidence_pack.items] == [[1], [2]]
    assert [item.query_ids for item in result.evidence_pack.items] == [["Q1"], ["Q4"]]


@pytest.mark.asyncio
async def test_first_round_missing_set_includes_aspects_without_supplement_queries() -> None:
    chunk = make_chunk("chunk-1", "现金退款有据可查。")

    async def executor(**kwargs: Any) -> RetrieveOnceResult:
        return RetrieveOnceResult(
            query_id=kwargs["query_id"], query=kwargs["search_query"],
            aspect_ids=kwargs["aspect_ids"], round=kwargs["round"],
            candidate_snapshot=[], retrieved_chunks=[retrieved(chunk)],
            metadata={}, latency_ms=1,
        )

    service, _, _, _ = build_service(
        fast_outputs=[
            {"aspects": ["现金退款", "积分退回", "积分有效期"],
             "queries": [{"query": "退款规则", "aspect_indexes": [0, 1, 2]}]},
            {"aspects": [
                {"aspect_index": 0, "evidence": [{"candidate_index": 0, "role": "core"}]},
                {"aspect_index": 1, "evidence": []},
                {"aspect_index": 2, "evidence": []},
            ], "decision": "supplement",
             "next_queries": [{"query": "积分有效期规则", "aspect_indexes": [2]}]},
        ],
        strong_outputs=[{
            "aspects": [
                {"aspect_index": 0, "evidence": [{"candidate_index": 0, "role": "core"}]},
                {"aspect_index": 1, "evidence": []},
                {"aspect_index": 2, "evidence": []},
            ],
            "answer_blocks": [
                {"text": "现金退款有据可查。", "evidence_indexes": [0], "kind": "supported"},
                {"text": "积分规则证据不足。", "evidence_indexes": [], "kind": "insufficient"},
            ],
        }],
        chunks=[chunk], executor=executor,
    )
    result = await service.research(
        ResearchRequest(kb_id="kb-a", user_id="user-a", query="退款规则？"),
        tenant_id="tenant-a", tenant=SimpleNamespace(plan="pro"),
    )

    assert result.metadata.first_round_missing_aspect_ids == ["A2", "A3"]
    assert result.rounds[1].tasks[0].query_id == "Q4"
    assert result.rounds[1].tasks[0].aspect_ids == ["A3"]
    assert result.evidence_pack.missing_aspects == ["积分退回", "积分有效期"]


@pytest.mark.asyncio
async def test_finalizer_failure_returns_decider_pack_without_answer() -> None:
    chunk = make_chunk("chunk-1", "退款积分按积分有效期规则处理。")

    async def executor(**kwargs: Any) -> RetrieveOnceResult:
        return RetrieveOnceResult(
            query_id=kwargs["query_id"],
            query=kwargs["search_query"],
            aspect_ids=kwargs["aspect_ids"],
            round=kwargs["round"],
            candidate_snapshot=[],
            retrieved_chunks=[retrieved(chunk)],
            metadata={},
            latency_ms=1,
        )

    service, _, _, _ = build_service(
        fast_outputs=[planner_output(), decider_output()],
        strong_outputs=[RuntimeError("provider unavailable")],
        chunks=[chunk],
        executor=executor,
    )

    result = await service.research(
        ResearchRequest(kb_id="kb-a", user_id="user-a", query="退款积分有效期？"),
        tenant_id="tenant-a",
        tenant=SimpleNamespace(plan="pro"),
    )

    assert result.answer is None
    assert result.evidence_pack.items[0].chunk_id == "chunk-1"
    assert result.metadata.stop_reason == "degraded"
    assert result.metadata.finalizer["degraded"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("denied_plan", ["free", "standard"])
async def test_research_permission_and_all_retrieval_failure_are_direct_errors(
    denied_plan: str,
) -> None:
    async def executor(**kwargs: Any) -> RetrieveOnceResult:
        del kwargs
        raise RuntimeError("retrieval unavailable")

    service, fast, _, rate_limit = build_service(
        fast_outputs=[planner_output()],
        strong_outputs=[],
        chunks=[],
        executor=executor,
    )

    with pytest.raises(AppError) as denied:
        await service.research(
            ResearchRequest(kb_id="kb-a", user_id="user-a", query="问题"),
            tenant_id="tenant-a",
            tenant=SimpleNamespace(plan=denied_plan),
        )
    assert denied.value.code == ErrorCode.FEATURE_NOT_ALLOWED.code
    assert fast.calls == []
    assert rate_limit.checks == 0

    with pytest.raises(AppError) as failed:
        await service.research(
            ResearchRequest(kb_id="kb-a", user_id="user-a", query="问题"),
            tenant_id="tenant-a",
            tenant=SimpleNamespace(plan="pro"),
        )
    assert failed.value.code == ErrorCode.RETRIEVAL_FAILED.code
    assert rate_limit.checks == 1
    assert rate_limit.records == 0


def test_research_request_rejects_retrieval_overrides() -> None:
    with pytest.raises(ValueError):
        ResearchRequest.model_validate(
            {
                "kb_id": "kb-a",
                "user_id": "user-a",
                "query": "问题",
                "profile": "quality",
                "max_rounds": 9,
            }
        )


@pytest.mark.asyncio
async def test_research_marks_successful_but_degraded_rerank_task() -> None:
    chunk = make_chunk("chunk-1", "积分按原规则处理。")

    async def executor(**kwargs: Any) -> RetrieveOnceResult:
        return RetrieveOnceResult(
            query_id=kwargs["query_id"], query=kwargs["search_query"],
            aspect_ids=kwargs["aspect_ids"], round=kwargs["round"],
            candidate_snapshot=[], retrieved_chunks=[retrieved(chunk)],
            metadata={"rerank": {"degraded": True, "error": "rerank HTTP 429"}},
            latency_ms=1, degraded=True,
        )

    service, _, _, _ = build_service(
        fast_outputs=[planner_output(), decider_output()],
        strong_outputs=[{
            "aspects": [{"aspect_index": 0, "evidence": [
                {"candidate_index": 0, "role": "core"}
            ]}],
            "answer_blocks": [{"text": "积分按原规则处理。", "evidence_indexes": [0],
                               "kind": "supported"}],
        }],
        chunks=[chunk], executor=executor,
    )
    result = await service.research(
        ResearchRequest(kb_id="kb-a", user_id="user-a", query="积分有效期？"),
        tenant_id="tenant-a", tenant=SimpleNamespace(plan="pro"),
    )
    assert result.answer == "积分按原规则处理。[E1]"
    assert result.rounds[0].tasks[0].success is True
    assert result.rounds[0].tasks[0].degraded is True
    assert result.metadata.degraded is True
    assert result.metadata.stop_reason == "degraded"


@pytest.mark.asyncio
async def test_decider_timeout_keeps_unverified_pack_empty_but_finalizer_can_recover() -> None:
    chunk = make_chunk("chunk-1", "积分按原规则处理。")

    async def executor(**kwargs: Any) -> RetrieveOnceResult:
        return RetrieveOnceResult(
            query_id=kwargs["query_id"], query=kwargs["search_query"],
            aspect_ids=kwargs["aspect_ids"], round=kwargs["round"],
            candidate_snapshot=[], retrieved_chunks=[retrieved(chunk)],
            metadata={}, latency_ms=1,
        )

    service, _, _, _ = build_service(
        fast_outputs=[planner_output(), LLMServiceError(code=ErrorCode.LLM_TIMEOUT)],
        strong_outputs=[{
            "aspects": [{"aspect_index": 0, "evidence": [
                {"candidate_index": 0, "role": "core"}
            ]}],
            "answer_blocks": [{"text": "积分按原规则处理。", "evidence_indexes": [0],
                               "kind": "supported"}],
        }],
        chunks=[chunk], executor=executor,
    )
    result = await service.research(
        ResearchRequest(kb_id="kb-a", user_id="user-a", query="积分有效期？"),
        tenant_id="tenant-a", tenant=SimpleNamespace(plan="pro"),
    )
    assert result.metadata.decider["degraded"] is True
    assert result.metadata.first_round_missing_aspect_ids == ["A1"]
    assert result.answer == "积分按原规则处理。[E1]"
    assert result.evidence_pack.status == "complete"
    assert result.metadata.stop_reason == "timeout"


@pytest.mark.asyncio
async def test_finalizer_timeout_returns_last_validated_pack() -> None:
    chunk = make_chunk("chunk-1", "积分按原规则处理。")

    async def executor(**kwargs: Any) -> RetrieveOnceResult:
        return RetrieveOnceResult(
            query_id=kwargs["query_id"], query=kwargs["search_query"],
            aspect_ids=kwargs["aspect_ids"], round=kwargs["round"],
            candidate_snapshot=[], retrieved_chunks=[retrieved(chunk)],
            metadata={}, latency_ms=1,
        )

    service, _, _, _ = build_service(
        fast_outputs=[planner_output(), decider_output()],
        strong_outputs=[TimeoutError()], chunks=[chunk], executor=executor,
    )
    result = await service.research(
        ResearchRequest(kb_id="kb-a", user_id="user-a", query="积分有效期？"),
        tenant_id="tenant-a", tenant=SimpleNamespace(plan="pro"),
    )
    assert result.answer is None
    assert result.evidence_pack.items[0].chunk_id == "chunk-1"
    assert result.metadata.finalizer["degraded"] is True
    assert result.metadata.stop_reason == "timeout"


@pytest.mark.asyncio
async def test_cancellation_is_not_swallowed_by_parallel_retrieval() -> None:
    from app.schemas.research import PlannedQuery, ResearchState
    from app.services.research_parallel_retrieval_service import ResearchParallelRetrievalService

    async def executor(**kwargs: Any) -> RetrieveOnceResult:
        del kwargs
        raise asyncio.CancelledError()

    state = ResearchState(
        research_id="r", original_query="问题", tenant_id="tenant-a",
        kb_ids=["kb-a"], index_versions={"kb-a": "v2"},
    )
    with pytest.raises(asyncio.CancelledError):
        await ResearchParallelRetrievalService(executor).execute(
            state=state,
            queries=[PlannedQuery(query_id="Q1", query="问题", aspect_ids=["A1"])],
            user_id="user-a", plan=PlanResolver().resolve("pro"),
            round_number=1, timeout_seconds=1,
        )


@pytest.mark.asyncio
async def test_index_version_is_frozen_even_if_active_version_changes_between_rounds() -> None:
    clue = make_chunk("clue", "有效期参考规则。")
    rule = make_chunk("rule", "退回积分按照剩余有效期计算。")
    clue.index_version = "v1"
    rule.index_version = "v1"
    kb = SimpleNamespace(id="kb-a", name="rules", description="", active_index_version="v1")

    class ChangingKnowledgeBaseRepository:
        async def get_by_ids(self, *, tenant_id, kb_ids):
            assert tenant_id == "tenant-a" and kb_ids == ["kb-a"]
            return [kb]

    seen_versions = []

    async def executor(**kwargs):
        seen_versions.append(dict(kwargs["index_versions"]))
        if kwargs["round"] == 1:
            kb.active_index_version = "v2"
        chunk = clue if kwargs["round"] == 1 else rule
        return RetrieveOnceResult(
            query_id=kwargs["query_id"], query=kwargs["search_query"],
            aspect_ids=kwargs["aspect_ids"], round=kwargs["round"],
            candidate_snapshot=[], retrieved_chunks=[retrieved(chunk)],
            metadata={}, latency_ms=1,
        )

    service, _, _, _ = build_service(
        fast_outputs=[planner_output(), decider_output(role="supporting", next_query=True)],
        strong_outputs=[{
            "aspects": [{"aspect_index": 0, "evidence": [
                {"candidate_index": 0, "role": "supporting"},
                {"candidate_index": 1, "role": "core"},
            ]}],
            "answer_blocks": [{"text": "按剩余有效期计算。", "evidence_indexes": [1],
                               "kind": "supported"}],
        }],
        chunks=[clue, rule], executor=executor,
    )
    service.knowledge_base_repository = ChangingKnowledgeBaseRepository()
    result = await service.research(
        ResearchRequest(kb_id="kb-a", user_id="user-a", query="退款积分有效期？"),
        tenant_id="tenant-a", tenant=SimpleNamespace(plan="pro"),
    )
    assert seen_versions == [{"kb-a": "v1"}, {"kb-a": "v1"}]
    assert kb.active_index_version == "v2"
    assert result.metadata.index_versions == {"kb-a": "v1"}
    assert [item.index_version for item in result.evidence_pack.items] == ["v1", "v1"]


@pytest.mark.asyncio
@pytest.mark.parametrize("duplicate_query", [True, False])
async def test_supplement_stops_on_duplicate_or_no_new_chunk(duplicate_query: bool) -> None:
    chunk = make_chunk("chunk-1", "有效期参考规则，尚缺少具体期限。")
    calls = []

    async def executor(**kwargs):
        calls.append(kwargs["query_id"])
        return RetrieveOnceResult(
            query_id=kwargs["query_id"], query=kwargs["search_query"],
            aspect_ids=kwargs["aspect_ids"], round=kwargs["round"],
            candidate_snapshot=[], retrieved_chunks=[retrieved(chunk)],
            metadata={}, latency_ms=1,
        )

    decider = decider_output(role="supporting", next_query=True)
    if duplicate_query:
        decider["next_queries"][0]["query"] = "退款积分有效期。"
    service, _, _, _ = build_service(
        fast_outputs=[planner_output(), decider],
        strong_outputs=[{
            "aspects": [{"aspect_index": 0, "evidence": [
                {"candidate_index": 0, "role": "supporting"}
            ]}],
            "answer_blocks": [{"text": "当前证据不足。", "evidence_indexes": [],
                               "kind": "insufficient"}],
        }],
        chunks=[chunk], executor=executor,
    )
    result = await service.research(
        ResearchRequest(kb_id="kb-a", user_id="user-a", query="退款积分有效期？"),
        tenant_id="tenant-a", tenant=SimpleNamespace(plan="pro"),
    )
    assert result.metadata.stop_reason == (
        "duplicate_query" if duplicate_query else "no_new_evidence"
    )
    assert result.metadata.round_count == (1 if duplicate_query else 2)
    assert calls == (["Q1"] if duplicate_query else ["Q1", "Q4"])
    assert result.evidence_pack.status == "missing"
    if not duplicate_query:
        assert result.rounds[1].new_chunk_count == 0
        assert result.rounds[1].tasks[0].new_chunk_count == 0


@pytest.mark.asyncio
async def test_stop_reason_matches_response_trace_and_safe_runtime_log(monkeypatch, caplog) -> None:
    import logging

    import app.services.research_service as research_module

    captured = {}

    class FakeObservability:
        trace_id = "trace-1"

        def __init__(self, **kwargs):
            captured["metadata"] = kwargs

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def span(self, *args, **kwargs):
            captured.setdefault("spans", []).append((args[0], kwargs))

        def finish(self, *, output):
            captured["output"] = output

    monkeypatch.setattr(research_module, "ResearchObservability", FakeObservability)
    chunk = make_chunk("chunk-1", "secret candidate body")

    async def executor(**kwargs):
        return RetrieveOnceResult(
            query_id=kwargs["query_id"], query=kwargs["search_query"],
            aspect_ids=kwargs["aspect_ids"], round=kwargs["round"],
            candidate_snapshot=[], retrieved_chunks=[retrieved(chunk)],
            metadata={}, latency_ms=1,
        )

    service, _, _, _ = build_service(
        fast_outputs=[planner_output(), decider_output()],
        strong_outputs=[{
            "aspects": [{"aspect_index": 0, "evidence": [
                {"candidate_index": 0, "role": "core"}
            ]}],
            "answer_blocks": [{"text": "按原规则处理。", "evidence_indexes": [0],
                               "kind": "supported"}],
        }],
        chunks=[chunk], executor=executor,
    )
    with caplog.at_level(logging.INFO):
        result = await service.research(
            ResearchRequest(kb_id="kb-a", user_id="user-a", query="secret question"),
            tenant_id="tenant-a", tenant=SimpleNamespace(plan="pro"),
        )
    assert captured["output"]["stop_reason"] == result.metadata.stop_reason
    assert f"stop_reason={result.metadata.stop_reason}" in caplog.text
    assert "secret candidate body" not in caplog.text
    assert "secret question" not in caplog.text
    assert captured["output"]["chunk_ids"] == ["chunk-1"]
    assert captured["output"]["input_tokens"] == 30
    assert captured["output"]["output_tokens"] == 15
    assert [name for name, _ in captured["spans"]] == [
        "planner", "retrieval_round_1", "retrieval_task:Q1",
        "evidence_decider", "finalizer",
    ]
    planner_span = captured["spans"][0][1]["output"]
    assert planner_span["model"] == "fast-model"
    assert planner_span["input_tokens"] == 10
    assert captured["spans"][2][1]["output"]["returned_count"] == 1


@pytest.mark.asyncio
async def test_slow_final_evidence_reload_marks_timeout_but_keeps_valid_answer(monkeypatch):
    import app.services.research_service as research_module

    monkeypatch.setattr(research_module, "DEFAULT_RESEARCH_TIMEOUT_SECONDS", 0.1)
    chunk = make_chunk("chunk-1", "积分按原规则处理。")

    class SlowSecondReload(FakeChunkRepository):
        def __init__(self):
            super().__init__([chunk])
            self.calls = 0

        async def get_by_scopes(self, **kwargs):
            self.calls += 1
            if self.calls == 2:
                await asyncio.sleep(0.12)
            return await super().get_by_scopes(**kwargs)

    async def executor(**kwargs):
        return RetrieveOnceResult(
            query_id=kwargs["query_id"], query=kwargs["search_query"],
            aspect_ids=kwargs["aspect_ids"], round=kwargs["round"],
            candidate_snapshot=[], retrieved_chunks=[retrieved(chunk)],
            metadata={}, latency_ms=1,
        )

    service, _, _, _ = build_service(
        fast_outputs=[planner_output(), decider_output()],
        strong_outputs=[{
            "aspects": [{"aspect_index": 0, "evidence": [
                {"candidate_index": 0, "role": "core"}
            ]}],
            "answer_blocks": [{"text": "按原规则处理。", "evidence_indexes": [0],
                               "kind": "supported"}],
        }],
        chunks=[chunk], executor=executor,
    )
    slow_repo = SlowSecondReload()
    service.evidence_service.chunk_repository = slow_repo
    result = await service.research(
        ResearchRequest(kb_id="kb-a", user_id="user-a", query="积分规则？"),
        tenant_id="tenant-a", tenant=SimpleNamespace(plan="pro"),
    )
    assert slow_repo.calls == 2
    assert result.answer == "按原规则处理。[E1]"
    assert result.metadata.latency_ms >= 100
    assert result.metadata.stop_reason == "timeout"
