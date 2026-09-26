import asyncio

import pytest

from app.schemas.rag import RetrievedChunk
from app.schemas.research import PlannedQuery, ResearchState
from app.services.research_parallel_retrieval_service import ResearchParallelRetrievalService
from app.services.retrieve_once_service import RetrieveOnceResult
from app.tenant.plan_resolver import PlanResolver


@pytest.mark.asyncio
async def test_parallel_tasks_are_isolated_and_partial_failure_keeps_successes() -> None:
    active = 0
    peak = 0
    started = asyncio.Event()
    calls: list[tuple[str, str, tuple[str, ...], dict[str, str]]] = []

    async def executor(**kwargs):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        calls.append((
            kwargs["query_id"], kwargs["search_query"], tuple(kwargs["kb_ids"]),
            dict(kwargs["index_versions"]),
        ))
        if active == 3:
            started.set()
        try:
            await started.wait()
            await asyncio.sleep(0.01)
            if kwargs["query_id"] == "Q2":
                raise RuntimeError("secret candidate content must not escape")
            return RetrieveOnceResult(
                query_id=kwargs["query_id"], query=kwargs["search_query"],
                aspect_ids=kwargs["aspect_ids"], round=kwargs["round"],
                candidate_snapshot=[], metadata={}, latency_ms=10,
                retrieved_chunks=[RetrievedChunk(
                    document_id="doc", chunk_id=kwargs["query_id"], kb_id="kb-a",
                    index_version="v1", title="title", content="content", score=0.5,
                )],
            )
        finally:
            active -= 1

    state = ResearchState(
        research_id="research", original_query="question", tenant_id="tenant-a",
        kb_ids=["kb-a", "kb-b"], index_versions={"kb-a": "v1", "kb-b": "v2"},
    )
    queries = [
        PlannedQuery(query_id=f"Q{i}", query=f"query {i}", aspect_ids=[f"A{i}"])
        for i in range(1, 4)
    ]
    result = await asyncio.wait_for(
        ResearchParallelRetrievalService(executor).execute(
            state=state, queries=queries, user_id="user-a",
            plan=PlanResolver().resolve("pro"), round_number=1, timeout_seconds=1,
        ),
        timeout=2,
    )

    assert peak == 3
    assert calls == [
        (f"Q{i}", f"query {i}", ("kb-a", "kb-b"), {"kb-a": "v1", "kb-b": "v2"})
        for i in range(1, 4)
    ]
    assert result.success_count == 2
    assert result.failed_count == 1
    assert result.degraded is True
    assert state.round_count == 1
    assert state.retrieval_task_count == 3
    assert [task.chunk_count for task in result.round.tasks] == [1, 0, 1]
    assert "secret candidate" not in str(result.round.model_dump())
    assert all("chunk_ids" not in task.model_dump() for task in result.round.tasks)
