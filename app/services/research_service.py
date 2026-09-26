from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Any

from app.core.config import Settings
from app.core.error_codes import ErrorCode
from app.core.exceptions import AppError, raise_app_error
from app.core.logging import get_logger
from app.observability.research_observability import ResearchObservability
from app.providers.llm.role_router import LLMRole, LLMRoleRouter
from app.repositories.chunk_repository import ChunkRepository
from app.repositories.knowledge_base_repository import KnowledgeBaseRepository
from app.schemas.research import (
    ResearchData,
    ResearchMetadata,
    ResearchRequest,
    ResearchState,
    ResearchStopReason,
)
from app.services.rate_limit_service import RateLimitService
from app.services.research_candidate_fusion_service import (
    ResearchCandidateFusionService,
)
from app.services.research_decider_service import ResearchDeciderService
from app.services.research_evidence_service import ResearchEvidenceService
from app.services.research_finalizer_service import ResearchFinalizerService
from app.services.research_limits import (
    DEFAULT_RESEARCH_TIMEOUT_SECONDS,
    MAX_RESEARCH_OUTPUT_TOKENS,
)
from app.services.research_parallel_retrieval_service import (
    ResearchParallelRetrievalService,
)
from app.services.research_planner_service import ResearchPlannerService
from app.services.retrieve_once_service import RetrieveOnceResult
from app.tenant.plan_resolver import PlanContext, PlanResolver
from app.utils.id_generator import generate_id

RetrieveExecutor = Callable[..., Awaitable[RetrieveOnceResult]]

_STOP_PRIORITY: dict[ResearchStopReason, int] = {
    "timeout": 7,
    "budget_exhausted": 6,
    "degraded": 5,
    "evidence_complete": 4,
    "duplicate_query": 3,
    "no_new_evidence": 2,
    "max_rounds": 1,
}


class ResearchService:
    def __init__(
        self,
        *,
        settings: Settings,
        knowledge_base_repository: KnowledgeBaseRepository,
        chunk_repository: ChunkRepository,
        plan_resolver: PlanResolver,
        rate_limit_service: RateLimitService | None,
        retrieve_executor: RetrieveExecutor,
        role_router: LLMRoleRouter | None = None,
    ) -> None:
        self.settings = settings
        self.knowledge_base_repository = knowledge_base_repository
        self.plan_resolver = plan_resolver
        self.rate_limit_service = rate_limit_service
        self.role_router = role_router or LLMRoleRouter(settings)
        self.fusion_service = ResearchCandidateFusionService()
        self.parallel_service = ResearchParallelRetrievalService(retrieve_executor)
        self.evidence_service = ResearchEvidenceService(chunk_repository)
        self.logger = get_logger(__name__)

    async def research(
        self,
        request: ResearchRequest,
        *,
        tenant_id: str,
        tenant: Any | None = None,
    ) -> ResearchData:
        started = time.perf_counter()
        deadline = started + DEFAULT_RESEARCH_TIMEOUT_SECONDS
        research_id = generate_id()
        log_id = generate_id()
        self._validate_query(request.query)
        kb_ids = request.resolved_kb_ids()
        plan = await self._resolve_plan(tenant_id, tenant=tenant)
        self._enforce_policy(plan, kb_ids=kb_ids)
        knowledge_bases = await self._load_knowledge_bases(
            tenant_id=tenant_id, kb_ids=kb_ids
        )
        index_versions = {
            str(kb.id): str(getattr(kb, "active_index_version", None) or "v1")
            for kb in knowledge_bases
        }
        if self.rate_limit_service is not None:
            await self.rate_limit_service.check_retrieve(tenant_id, plan)

        state = ResearchState(
            research_id=research_id,
            original_query=request.query,
            tenant_id=tenant_id,
            kb_ids=kb_ids,
            index_versions=index_versions,
        )
        fast_provider = self.role_router.provider_for(LLMRole.FAST)
        strong_model = self.role_router.model_for(LLMRole.STRONG)
        strong_fallback = not self.settings.llm_strong_model.strip()
        strong_provider = self.role_router.provider_for(LLMRole.STRONG)
        planner = ResearchPlannerService(
            fast_provider, model=self.role_router.model_for(LLMRole.FAST)
        )
        decider = ResearchDeciderService(
            fast_provider,
            self.evidence_service,
            model=self.role_router.model_for(LLMRole.FAST),
        )
        finalizer = ResearchFinalizerService(
            strong_provider,
            self.evidence_service,
            model=strong_model,
            strong_model_fallback=strong_fallback,
        )
        stop_conditions: list[ResearchStopReason] = []

        observability = ResearchObservability(
            settings=self.settings,
            research_id=research_id,
            log_id=log_id,
            tenant_id=tenant_id,
            user_id=request.user_id,
            tenant_plan=plan.plan,
            kb_ids=kb_ids,
            index_versions=index_versions,
            query=request.query,
        )
        with observability:
            state.stage = "planning"
            plan_data = await planner.plan(
                state=state,
                knowledge_bases=knowledge_bases,
                timeout_seconds=min(self._remaining(deadline), 10),
            )
            observability.span(
                "planner",
                output={
                    **self._stage_observation(state),
                    "aspect_count": len(plan_data.aspects),
                    "query_count": len(plan_data.initial_queries),
                    "degraded": plan_data.degraded,
                },
            )
            if plan_data.degraded:
                stop_conditions.append("degraded")

            state.stage = "retrieving"
            round_one = await self.parallel_service.execute(
                state=state,
                queries=plan_data.initial_queries,
                user_id=request.user_id,
                plan=plan,
                round_number=1,
                timeout_seconds=self._remaining(deadline),
            )
            self._observe_round(observability, round_one.round, round_one.results)
            if round_one.success_count == 0:
                if time.perf_counter() >= deadline:
                    raise_app_error(ErrorCode.API_TIMEOUT)
                raise AppError(code=ErrorCode.RETRIEVAL_FAILED)
            if round_one.degraded:
                state.degraded = True
                stop_conditions.append("degraded")

            candidates = self.fusion_service.merge(round_one.results)
            state.candidate_chunk_ids = [candidate.chunk_id for candidate in candidates]
            round_one.round.candidate_count = len(candidates)
            round_one.round.new_chunk_count = len(candidates)
            self._set_task_new_counts(round_one.round.tasks, round_one.results, existing_keys=set())

            state.stage = "deciding"
            decision = await decider.decide(
                state=state,
                candidates=candidates,
                timeout_seconds=min(self._remaining(deadline), 15),
            )
            first_round_missing_aspect_ids = [
                aspect.aspect_id
                for aspect, group in zip(state.aspects, decision.evidence_pack.groups, strict=True)
                if not group.covered
            ]
            observability.span(
                "evidence_decider",
                output={
                    **self._stage_observation(state),
                    "candidate_count": len(candidates),
                    "evidence_status": decision.evidence_pack.status,
                    "covered_aspect_count": sum(
                        group.covered for group in decision.evidence_pack.groups
                    ),
                    "missing_aspect_count": len(decision.evidence_pack.missing_aspects),
                    "next_query_count": len(decision.next_queries),
                    "degraded": decision.degraded,
                },
            )
            fallback_pack = decision.evidence_pack
            if decision.degraded:
                stop_conditions.append("degraded")
            elif decision.evidence_pack.status == "complete":
                stop_conditions.append("evidence_complete")
            elif decision.next_queries:
                state.stage = "supplementing"
                old_keys = {
                    (candidate.kb_id, candidate.index_version, candidate.chunk_id)
                    for candidate in candidates
                }
                round_two = await self.parallel_service.execute(
                    state=state,
                    queries=decision.next_queries,
                    user_id=request.user_id,
                    plan=plan,
                    round_number=2,
                    timeout_seconds=self._remaining(deadline),
                )
                self._observe_round(observability, round_two.round, round_two.results)
                if round_two.success_count == 0:
                    state.degraded = True
                    stop_conditions.append("degraded")
                merged = self.fusion_service.merge([*round_one.results, *round_two.results])
                new_count = sum(
                    (candidate.kb_id, candidate.index_version, candidate.chunk_id)
                    not in old_keys
                    for candidate in merged
                )
                round_two.round.candidate_count = len(merged)
                round_two.round.new_chunk_count = new_count
                self._set_task_new_counts(
                    round_two.round.tasks,
                    round_two.results,
                    existing_keys=self._retrieved_chunk_keys(round_one.results),
                )
                candidates = merged
                state.candidate_chunk_ids = [candidate.chunk_id for candidate in candidates]
                stop_conditions.append("no_new_evidence" if new_count == 0 else "max_rounds")
            elif decision.duplicate_queries:
                stop_conditions.append("duplicate_query")
            else:
                state.degraded = True
                stop_conditions.append("degraded")

            if self._remaining(deadline) <= 0:
                stop_conditions.append("timeout")
                raise_app_error(ErrorCode.API_TIMEOUT)
            state.stage = "finalizing"
            final = await finalizer.finalize(
                state=state,
                candidates=candidates,
                fallback_pack=fallback_pack,
                timeout_seconds=min(
                    self._remaining(deadline),
                    self.settings.llm_timeout_seconds,
                ),
            )
            observability.span(
                "finalizer",
                output={
                    **self._stage_observation(state),
                    "evidence_count": len(final.evidence_pack.items),
                    "answer_chars": len(final.answer or ""),
                    "status": final.evidence_pack.status,
                    "degraded": final.degraded,
                },
            )
            if final.degraded:
                stop_conditions.append("degraded")
            elif final.evidence_pack.status == "complete":
                stop_conditions.append("evidence_complete")
            if time.perf_counter() >= deadline:
                stop_conditions.append("timeout")
            if any(stage.error and "timeout" in stage.error.lower() for stage in state.stages):
                stop_conditions.append("timeout")
            if any(
                task.error and "timeout" in task.error.lower()
                for round_data in state.rounds
                for task in round_data.tasks
            ):
                stop_conditions.append("timeout")
            if state.output_tokens > MAX_RESEARCH_OUTPUT_TOKENS or any(
                "budget exhausted" in error for error in state.errors
            ):
                stop_conditions.append("budget_exhausted")

            state.stage = "completed"
            state.elapsed_ms = int((time.perf_counter() - started) * 1000)
            stop_reason = self._resolve_stop_reason(stop_conditions)
            state.stop_reason = stop_reason
            if self.rate_limit_service is not None:
                await self.rate_limit_service.record_retrieve_success(tenant_id)
            metadata = ResearchMetadata(
                research_id=research_id,
                log_id=log_id,
                trace_id=observability.trace_id,
                tenant_plan=plan.plan,
                index_versions=index_versions,
                round_count=state.round_count,
                retrieval_task_count=state.retrieval_task_count,
                llm_call_count=state.llm_call_count,
                input_tokens=state.input_tokens,
                output_tokens=state.output_tokens,
                latency_ms=state.elapsed_ms,
                stop_reason=stop_reason,
                first_round_missing_aspect_ids=first_round_missing_aspect_ids,
                degraded=state.degraded,
                errors=state.errors,
                stages=state.stages,
                planner={
                    "degraded": plan_data.degraded,
                    "error": plan_data.error,
                },
                decider={
                    "degraded": decision.degraded,
                    "error": decision.error,
                },
                finalizer={
                    "degraded": final.degraded,
                    "error": final.error,
                    "strong_model_fallback": strong_fallback,
                },
            )
            response = ResearchData(
                query=request.query,
                kb_id=kb_ids[0],
                kb_ids=kb_ids,
                answer=final.answer,
                evidence_pack=final.evidence_pack,
                plan=plan_data,
                rounds=state.rounds,
                metadata=metadata,
            )
            observability.finish(
                output={
                    "evidence_status": final.evidence_pack.status,
                    "evidence_ids": [item.evidence_id for item in final.evidence_pack.items],
                    "chunk_ids": [item.chunk_id for item in final.evidence_pack.items],
                    "round_count": state.round_count,
                    "retrieval_task_count": state.retrieval_task_count,
                    "llm_call_count": state.llm_call_count,
                    "input_tokens": state.input_tokens,
                    "output_tokens": state.output_tokens,
                    "stop_reason": stop_reason,
                    "latency_ms": state.elapsed_ms,
                    "degraded": state.degraded,
                }
            )
            self.logger.info(
                "RESEARCH_COMPLETE research_id=%s stop_reason=%s round_count=%d "
                "retrieval_task_count=%d llm_call_count=%d degraded=%s",
                research_id,
                stop_reason,
                state.round_count,
                state.retrieval_task_count,
                state.llm_call_count,
                state.degraded,
            )
            return response

    def _validate_query(self, query: str) -> None:
        if not query.strip():
            raise_app_error(ErrorCode.PARAM_ERROR, "query must not be blank")
        if len(query.strip()) > self.settings.query_max_length:
            raise_app_error(
                ErrorCode.PARAM_ERROR,
                f"query must not exceed {self.settings.query_max_length} characters",
            )

    async def _resolve_plan(
        self, tenant_id: str, *, tenant: Any | None
    ) -> PlanContext:
        if tenant is not None:
            return self.plan_resolver.resolve(tenant)
        return await self.plan_resolver.resolve_for_tenant_id(tenant_id)

    @staticmethod
    def _enforce_policy(plan: PlanContext, *, kb_ids: list[str]) -> None:
        if not plan.features.research_allowed:
            raise_app_error(
                ErrorCode.FEATURE_NOT_ALLOWED,
                data={"plan": plan.plan, "feature": "research"},
            )
        if len(kb_ids) > plan.limits.max_kb_per_retrieve:
            raise_app_error(
                ErrorCode.FEATURE_NOT_ALLOWED,
                data={
                    "plan": plan.plan,
                    "feature": "multi-knowledge-base research",
                    "max_kb_per_retrieve": plan.limits.max_kb_per_retrieve,
                },
            )

    async def _load_knowledge_bases(
        self, *, tenant_id: str, kb_ids: list[str]
    ) -> list[Any]:
        items = await self.knowledge_base_repository.get_by_ids(
            tenant_id=tenant_id, kb_ids=kb_ids
        )
        by_id = {str(item.id): item for item in items}
        missing = [kb_id for kb_id in kb_ids if kb_id not in by_id]
        if missing:
            raise_app_error(
                ErrorCode.NOT_FOUND,
                "knowledge base not found",
                data={"missing_kb_ids": missing},
            )
        return [by_id[kb_id] for kb_id in kb_ids]

    @staticmethod
    def _remaining(deadline: float) -> float:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise_app_error(ErrorCode.API_TIMEOUT)
        return remaining

    @staticmethod
    def _resolve_stop_reason(
        conditions: list[ResearchStopReason],
    ) -> ResearchStopReason:
        if not conditions:
            return "max_rounds"
        return max(conditions, key=_STOP_PRIORITY.__getitem__)

    @staticmethod
    def _stage_observation(state: ResearchState) -> dict[str, Any]:
        stage = state.stages[-1]
        return {
            "model": stage.model,
            "role": stage.role,
            "latency_ms": stage.latency_ms,
            "input_tokens": stage.input_tokens,
            "output_tokens": stage.output_tokens,
            "degraded": stage.degraded,
        }

    @staticmethod
    def _retrieved_chunk_keys(
        results: list[RetrieveOnceResult],
    ) -> set[tuple[str, str, str]]:
        return {
            (chunk.kb_id, chunk.index_version, chunk.chunk_id)
            for result in results
            for chunk in result.retrieved_chunks
        }

    @staticmethod
    def _set_task_new_counts(
        tasks: list[Any],
        results: list[RetrieveOnceResult],
        *,
        existing_keys: set[tuple[str, str, str]],
    ) -> None:
        seen = set(existing_keys)
        for task, result in zip(tasks, results, strict=True):
            keys = {
                (chunk.kb_id, chunk.index_version, chunk.chunk_id)
                for chunk in result.retrieved_chunks
            }
            task.new_chunk_count = len(keys - seen)
            seen.update(keys)

    @staticmethod
    def _observe_round(
        observability: ResearchObservability,
        round_data: Any,
        results: list[RetrieveOnceResult],
    ) -> None:
        name = f"retrieval_round_{round_data.round}"
        observability.span(
            name,
            output={
                "purpose": round_data.purpose,
                "task_count": len(round_data.tasks),
                "latency_ms": round_data.latency_ms,
            },
        )
        for task, result in zip(round_data.tasks, results, strict=True):
            observability.span(
                f"retrieval_task:{task.query_id}",
                input={
                    "round": task.round,
                    "query_id": task.query_id,
                    "aspect_ids": task.aspect_ids,
                },
                output={
                    "success": task.success,
                    "retrieval_mode": "hybrid",
                    "candidate_count": len(result.candidate_snapshot),
                    "returned_count": len(result.retrieved_chunks),
                    "chunk_ids": [chunk.chunk_id for chunk in result.retrieved_chunks],
                    "latency_ms": task.latency_ms,
                    "degraded": task.degraded,
                },
            )
