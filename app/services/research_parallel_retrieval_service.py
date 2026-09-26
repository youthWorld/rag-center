from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from app.schemas.research import PlannedQuery, ResearchRound, ResearchState, ResearchTaskResult
from app.services.research_common import safe_stage_error
from app.services.research_limits import MAX_INITIAL_QUERIES, MAX_SUPPLEMENT_QUERIES
from app.services.retrieve_once_service import RetrieveOnceResult
from app.tenant.plan_resolver import PlanContext

RetrieveExecutor = Callable[..., Awaitable[RetrieveOnceResult]]


@dataclass(slots=True)
class ParallelRetrievalResult:
    results: list[RetrieveOnceResult]
    round: ResearchRound
    success_count: int
    failed_count: int
    latency_ms: int
    degraded: bool


class ResearchParallelRetrievalService:
    def __init__(self, executor: RetrieveExecutor) -> None:
        self.executor = executor

    async def execute(
        self,
        *,
        state: ResearchState,
        queries: list[PlannedQuery],
        user_id: str,
        plan: PlanContext,
        round_number: int,
        timeout_seconds: float,
    ) -> ParallelRetrievalResult:
        limit = MAX_INITIAL_QUERIES if round_number == 1 else MAX_SUPPLEMENT_QUERIES
        selected = queries[:limit]
        semaphore = asyncio.Semaphore(limit)
        started = time.perf_counter()

        async def run(query: PlannedQuery) -> RetrieveOnceResult:
            state.mark_visited(query.query)
            task_started = time.perf_counter()
            try:
                async with semaphore:
                    return await asyncio.wait_for(
                        self.executor(
                            tenant_id=state.tenant_id,
                            user_id=user_id,
                            kb_ids=state.kb_ids,
                            index_versions=state.index_versions,
                            query_id=query.query_id,
                            search_query=query.query,
                            aspect_ids=query.aspect_ids,
                            round=round_number,
                            plan=plan,
                        ),
                        timeout=timeout_seconds,
                    )
            except Exception as exception:
                return RetrieveOnceResult(
                    query_id=query.query_id,
                    query=query.query,
                    aspect_ids=query.aspect_ids,
                    round=round_number,
                    candidate_snapshot=[],
                    retrieved_chunks=[],
                    metadata={},
                    latency_ms=int((time.perf_counter() - task_started) * 1000),
                    degraded=True,
                    error=safe_stage_error("retrieval task", exception),
                )

        results = await asyncio.gather(*(run(query) for query in selected))
        latency_ms = int((time.perf_counter() - started) * 1000)
        tasks = [
            ResearchTaskResult(
                query_id=result.query_id,
                query=result.query,
                aspect_ids=result.aspect_ids,
                round=round_number,
                success=result.error is None,
                latency_ms=result.latency_ms,
                chunk_count=len(result.retrieved_chunks),
                degraded=result.degraded,
                error=result.error,
            )
            for result in results
        ]
        success_count = sum(task.success for task in tasks)
        round_data = ResearchRound(
            round=round_number,
            purpose="initial" if round_number == 1 else "supplement",
            tasks=tasks,
            candidate_count=sum(len(result.candidate_snapshot) for result in results),
            latency_ms=latency_ms,
        )
        state.rounds.append(round_data)
        state.round_count = max(state.round_count, round_number)
        state.retrieval_task_count += len(selected)
        return ParallelRetrievalResult(
            results=results,
            round=round_data,
            success_count=success_count,
            failed_count=len(tasks) - success_count,
            latency_ms=latency_ms,
            degraded=any(task.degraded for task in tasks),
        )
