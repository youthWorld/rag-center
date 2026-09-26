from __future__ import annotations

import asyncio
import time
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from app.providers.llm.base import LLMProvider
from app.providers.llm.role_router import LLMRole
from app.schemas.research import (
    PlannedQuery,
    ResearchAspect,
    ResearchPlanData,
    ResearchStageUsage,
    ResearchState,
    normalize_research_query,
)
from app.services.research_common import (
    ResearchValidationError,
    ensure_payload_budget,
    safe_stage_error,
    validate_search_query,
)
from app.services.research_limits import (
    MAX_ASPECT_CHARS,
    MAX_ASPECTS,
    MAX_INITIAL_QUERIES,
    MAX_RESEARCH_LLM_CALLS,
    PLANNER_MAX_TOKENS,
)

RESEARCH_PLANNER_PROMPT = """你是 RAG Research 的查询规划器，不负责回答问题。
识别回答问题必须覆盖的最少要点，并生成 1 到 3 个首轮检索查询；简单问题只生成一个查询。
严格只返回以下 JSON 对象结构（例子中的文字不是答案）：
{"aspects":["要点一","要点二"],
 "queries":[{"query":"针对要点一的检索句","aspect_indexes":[0]},
            {"query":"针对要点二的检索句","aspect_indexes":[1]}]}
aspects 必须是字符串数组；queries 必须是对象数组，不能是字符串数组。
aspect_indexes 使用从 0 开始的整数下标，所有要点必须至少被一个查询覆盖。
不得选择租户、知识库、索引、工具或生成答案。不得输出额外字段。"""


class PlannerQueryOutput(BaseModel):
    query: str
    aspect_indexes: list[int] = Field(min_length=1)


class PlannerOutput(BaseModel):
    aspects: list[str] = Field(min_length=1)
    queries: list[PlannerQueryOutput] = Field(min_length=1)


class ResearchPlannerService:
    def __init__(self, provider: LLMProvider, *, model: str) -> None:
        self.provider = provider
        self.model = model

    async def plan(
        self,
        *,
        state: ResearchState,
        knowledge_bases: list[Any],
        timeout_seconds: float,
    ) -> ResearchPlanData:
        started = time.perf_counter()
        degraded = False
        error: str | None = None
        input_tokens = 0
        output_tokens = 0
        try:
            if state.llm_call_count >= MAX_RESEARCH_LLM_CALLS:
                raise ResearchValidationError("research LLM call budget exhausted")
            payload = {
                "question": state.original_query,
                "knowledge_bases": [
                    {
                        "name": str(getattr(kb, "name", ""))[:255],
                        "description": str(getattr(kb, "description", "") or "")[:500],
                    }
                    for kb in knowledge_bases
                ],
                "limits": {
                    "max_aspects": MAX_ASPECTS,
                    "max_queries": MAX_INITIAL_QUERIES,
                    "max_query_chars": 300,
                },
            }
            ensure_payload_budget(payload)
            state.record_llm_attempt()
            result = await asyncio.wait_for(
                self.provider.chat_json_with_metadata(
                    system_prompt=RESEARCH_PLANNER_PROMPT,
                    user_payload=payload,
                    temperature=0.0,
                    timeout_seconds=timeout_seconds,
                    log_payload=False,
                    max_tokens=PLANNER_MAX_TOKENS,
                    enable_thinking=False,
                ),
                timeout=timeout_seconds,
            )
            input_tokens = int(result.metadata.input_tokens or 0)
            output_tokens = int(result.metadata.output_tokens or 0)
            state.record_usage(input_tokens=input_tokens, output_tokens=output_tokens)
            aspects, queries = self._validate(result.output)
        except Exception as exception:
            degraded = True
            error = safe_stage_error("planner", exception)
            aspects = [ResearchAspect(aspect_id="A1", description="完整回答用户问题")]
            queries = [
                PlannedQuery(
                    query_id="Q1",
                    query=state.original_query,
                    aspect_ids=["A1"],
                )
            ]
        usage = ResearchStageUsage(
            stage="planner",
            role=LLMRole.FAST,
            model=self.model,
            latency_ms=int((time.perf_counter() - started) * 1000),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            degraded=degraded,
            error=error,
        )
        state.stages.append(usage)
        state.aspects = aspects
        state.initial_queries = queries
        if degraded:
            state.degraded = True
            state.errors.append(error or "planner degraded")
        return ResearchPlanData(
            aspects=aspects,
            initial_queries=queries,
            degraded=degraded,
            error=error,
        )

    @staticmethod
    def _validate(payload: dict[str, Any]) -> tuple[list[ResearchAspect], list[PlannedQuery]]:
        try:
            raw = PlannerOutput.model_validate(payload)
        except ValidationError as exception:
            raise ResearchValidationError("invalid planner output") from exception
        if len(raw.aspects) > MAX_ASPECTS or len(raw.queries) > MAX_INITIAL_QUERIES:
            raise ResearchValidationError("planner output exceeds limits")
        aspects: list[ResearchAspect] = []
        aspect_keys: set[str] = set()
        for index, text in enumerate(raw.aspects):
            normalized = " ".join(text.strip().split())
            key = normalized.lower()
            if not normalized or len(normalized) > MAX_ASPECT_CHARS or key in aspect_keys:
                raise ResearchValidationError("invalid or duplicate aspect")
            aspect_keys.add(key)
            aspects.append(ResearchAspect(aspect_id=f"A{index + 1}", description=normalized))
        queries: list[PlannedQuery] = []
        query_keys: set[str] = set()
        covered: set[int] = set()
        for index, item in enumerate(raw.queries):
            query = validate_search_query(item.query)
            key = normalize_research_query(query)
            if key in query_keys:
                raise ResearchValidationError("duplicate planner query")
            indexes = item.aspect_indexes
            if len(indexes) != len(set(indexes)) or any(
                isinstance(value, bool) or value < 0 or value >= len(aspects) for value in indexes
            ):
                raise ResearchValidationError("invalid planner aspect index")
            query_keys.add(key)
            covered.update(indexes)
            queries.append(
                PlannedQuery(
                    query_id=f"Q{index + 1}",
                    query=query,
                    aspect_ids=[aspects[value].aspect_id for value in indexes],
                )
            )
        if covered != set(range(len(aspects))):
            raise ResearchValidationError("planner queries do not cover every aspect")
        return aspects, queries
