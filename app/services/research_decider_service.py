from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from app.providers.llm.base import LLMProvider
from app.providers.llm.role_router import LLMRole
from app.schemas.rag import EvidencePack
from app.schemas.research import (
    PlannedQuery,
    ResearchCandidate,
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
from app.services.research_evidence_service import (
    ResearchEvidenceService,
    empty_evidence_pack,
)
from app.services.research_limits import (
    DECIDER_MAX_TOKENS,
    MAX_CANDIDATE_TEXT_CHARS,
    MAX_RESEARCH_LLM_CALLS,
    MAX_SUPPLEMENT_QUERIES,
)

RESEARCH_DECIDER_PROMPT = """你是 RAG Research 的证据判断器，不负责生成最终回答。
为每个必要要点选择给定候选中的 core 或 supporting 证据，只有 core 能覆盖要点。
严格只返回以下 JSON 对象结构（下标从 0 开始，示例不是答案）：
{"aspects":[{"aspect_index":0,"evidence":[{"candidate_index":0,"role":"core"}]},
            {"aspect_index":1,"evidence":[]}],
 "decision":"supplement",
 "next_queries":[{"query":"针对缺失要点的检索句","aspect_indexes":[1]}]}
每个输入要点恰好有一个 aspects 条目；evidence 可为空。candidate_index 只能引用给定候选的 index。
若全部有 core，decision 为 stop 且 next_queries 为空。
若缺证，可生成最多两个只绑定缺失要点的定向查询。
不得输出 Chunk ID、正文、知识库、索引、工具或答案。不得输出额外字段。"""


class DeciderEvidenceOutput(BaseModel):
    candidate_index: int
    role: Literal["core", "supporting"]


class DeciderAspectOutput(BaseModel):
    aspect_index: int
    evidence: list[DeciderEvidenceOutput] = Field(default_factory=list)


class DeciderQueryOutput(BaseModel):
    query: str
    aspect_indexes: list[int] = Field(min_length=1)


class DeciderOutput(BaseModel):
    aspects: list[DeciderAspectOutput]
    decision: Literal["stop", "supplement"]
    next_queries: list[DeciderQueryOutput] = Field(default_factory=list)


@dataclass(slots=True)
class DeciderResult:
    evidence_pack: EvidencePack
    next_queries: list[PlannedQuery]
    degraded: bool = False
    error: str | None = None
    duplicate_queries: bool = False


class ResearchDeciderService:
    def __init__(
        self,
        provider: LLMProvider,
        evidence_service: ResearchEvidenceService,
        *,
        model: str,
    ) -> None:
        self.provider = provider
        self.evidence_service = evidence_service
        self.model = model

    async def decide(
        self,
        *,
        state: ResearchState,
        candidates: list[ResearchCandidate],
        timeout_seconds: float,
    ) -> DeciderResult:
        started = time.perf_counter()
        input_tokens = 0
        output_tokens = 0
        try:
            if state.llm_call_count >= MAX_RESEARCH_LLM_CALLS:
                raise ResearchValidationError("research LLM call budget exhausted")
            payload = self._build_payload(state, candidates)
            ensure_payload_budget(payload)
            state.record_llm_attempt()
            result = await asyncio.wait_for(
                self.provider.chat_json_with_metadata(
                    system_prompt=RESEARCH_DECIDER_PROMPT,
                    user_payload=payload,
                    temperature=0.0,
                    timeout_seconds=timeout_seconds,
                    log_payload=False,
                    max_tokens=DECIDER_MAX_TOKENS,
                    enable_thinking=False,
                ),
                timeout=timeout_seconds,
            )
            input_tokens = int(result.metadata.input_tokens or 0)
            output_tokens = int(result.metadata.output_tokens or 0)
            state.record_usage(input_tokens=input_tokens, output_tokens=output_tokens)
            selections, raw_next = self._validate_output(
                result.output, aspect_count=len(state.aspects), candidate_count=len(candidates)
            )
            validated = await self.evidence_service.build_pack(
                tenant_id=state.tenant_id,
                index_versions=state.index_versions,
                aspects=state.aspects,
                candidates=candidates,
                selections=selections,
            )
            missing_ids = {
                state.aspects[index].aspect_id
                for index, references in enumerate(selections)
                if not any(reference["role"] == "core" for reference in references)
            }
            next_queries, duplicate_queries = self._validate_next_queries(
                raw_next,
                state=state,
                missing_ids=missing_ids,
            )
            if validated.pack.status == "complete":
                next_queries = []
            usage = ResearchStageUsage(
                stage="decider",
                role=LLMRole.FAST,
                model=self.model,
                latency_ms=int((time.perf_counter() - started) * 1000),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
            state.stages.append(usage)
            state.missing_aspect_ids = sorted(missing_ids)
            return DeciderResult(
                evidence_pack=validated.pack,
                next_queries=next_queries,
                duplicate_queries=duplicate_queries,
            )
        except Exception as exception:
            error = safe_stage_error("decider", exception)
            state.degraded = True
            state.errors.append(error)
            state.missing_aspect_ids = [aspect.aspect_id for aspect in state.aspects]
            state.stages.append(
                ResearchStageUsage(
                    stage="decider",
                    role=LLMRole.FAST,
                    model=self.model,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    degraded=True,
                    error=error,
                )
            )
            return DeciderResult(
                evidence_pack=empty_evidence_pack(state.aspects),
                next_queries=[],
                degraded=True,
                error=error,
            )

    @staticmethod
    def _build_payload(
        state: ResearchState, candidates: list[ResearchCandidate]
    ) -> dict[str, Any]:
        return {
            "question": state.original_query,
            "aspects": [
                {"index": index, "aspect": aspect.description}
                for index, aspect in enumerate(state.aspects)
            ],
            "visited_queries": state.visited_queries,
            "candidates": [
                {
                    "index": index,
                    "title": candidate.title,
                    "heading_path": candidate.heading_path,
                    "text": (candidate.retrieval_text or candidate.content)[
                        :MAX_CANDIDATE_TEXT_CHARS
                    ],
                    "aspect_ids": candidate.aspect_ids,
                    "query_ids": candidate.query_ids,
                    "source": candidate.source,
                }
                for index, candidate in enumerate(candidates)
            ],
            "limits": {"max_next_queries": MAX_SUPPLEMENT_QUERIES},
        }

    @staticmethod
    def _validate_output(
        payload: dict[str, Any], *, aspect_count: int, candidate_count: int
    ) -> tuple[list[list[dict[str, Any]]], list[DeciderQueryOutput]]:
        try:
            output = DeciderOutput.model_validate(payload)
        except ValidationError as exception:
            raise ResearchValidationError("invalid decider output") from exception
        if len(output.aspects) != aspect_count:
            raise ResearchValidationError("decider aspect count mismatch")
        selections: list[list[dict[str, Any]] | None] = [None] * aspect_count
        for aspect in output.aspects:
            if (
                isinstance(aspect.aspect_index, bool)
                or aspect.aspect_index < 0
                or aspect.aspect_index >= aspect_count
                or selections[aspect.aspect_index] is not None
            ):
                raise ResearchValidationError("invalid or duplicate decider aspect")
            references: list[dict[str, Any]] = []
            seen: set[int] = set()
            for reference in aspect.evidence:
                index = reference.candidate_index
                if (
                    isinstance(index, bool)
                    or index < 0
                    or index >= candidate_count
                    or index in seen
                ):
                    raise ResearchValidationError("invalid decider candidate index")
                seen.add(index)
                references.append(
                    {"candidate_index": index, "role": reference.role}
                )
            selections[aspect.aspect_index] = references
        if any(item is None for item in selections):
            raise ResearchValidationError("decider omitted an aspect")
        if len(output.next_queries) > MAX_SUPPLEMENT_QUERIES:
            raise ResearchValidationError("too many supplement queries")
        return [item or [] for item in selections], output.next_queries

    @staticmethod
    def _validate_next_queries(
        raw_queries: list[DeciderQueryOutput],
        *,
        state: ResearchState,
        missing_ids: set[str],
    ) -> tuple[list[PlannedQuery], bool]:
        result: list[PlannedQuery] = []
        seen: set[str] = set()
        duplicate_queries = False
        for item in raw_queries:
            query = validate_search_query(item.query)
            key = normalize_research_query(query)
            if key in state.visited_queries or key in seen:
                duplicate_queries = True
                continue
            indexes = item.aspect_indexes
            if len(indexes) != len(set(indexes)) or any(
                isinstance(value, bool)
                or value < 0
                or value >= len(state.aspects)
                for value in indexes
            ):
                raise ResearchValidationError("invalid supplement aspect index")
            aspect_ids = [state.aspects[value].aspect_id for value in indexes]
            if not aspect_ids or any(aspect_id not in missing_ids for aspect_id in aspect_ids):
                raise ResearchValidationError("supplement query targets a covered aspect")
            seen.add(key)
            result.append(
                PlannedQuery(
                    query_id=f"Q{4 + len(result)}",
                    query=query,
                    aspect_ids=aspect_ids,
                )
            )
        return result, duplicate_queries
