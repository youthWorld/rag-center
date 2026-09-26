from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from app.providers.llm.base import LLMProvider
from app.providers.llm.role_router import LLMRole
from app.schemas.rag import EvidencePack
from app.schemas.research import ResearchCandidate, ResearchStageUsage, ResearchState
from app.services.research_common import (
    ResearchValidationError,
    ensure_payload_budget,
    safe_stage_error,
)
from app.services.research_evidence_service import ResearchEvidenceService
from app.services.research_limits import (
    FINALIZER_MAX_TOKENS,
    MAX_CANDIDATE_TEXT_CHARS,
    MAX_FINAL_ANSWER_CHARS,
    MAX_RESEARCH_LLM_CALLS,
)

RESEARCH_FINALIZER_PROMPT = """你是 RAG Research 的最终回答生成器，只能使用给定候选。
为每个必要要点选择 core/supporting 证据并生成简洁回答块。
严格只返回以下 JSON 对象结构（下标从 0 开始，示例不是答案）：
{"aspects":[{"aspect_index":0,"evidence":[{"candidate_index":0,"role":"core"}]},
            {"aspect_index":1,"evidence":[]}],
 "answer_blocks":[{"text":"有原文支持的简洁结论。","evidence_indexes":[0],"kind":"supported"},
                  {"text":"该要点材料不足。","evidence_indexes":[],"kind":"insufficient"}]}
每个输入要点恰好有一个 aspects 条目。candidate_index 只能引用给定候选的 index。
supported 块引用的下标必须已在 aspects.evidence 中选择，不能缺少证据；insufficient 块不能引用证据。
证据不足时说明材料不足；冲突时说明来源不一致。至少返回一个 answer_block。
不要在 text 中输出 Markdown 引用、Chunk ID、候选编号或候选外事实。
引用由后端生成。不得输出额外字段。"""

_UNRESOLVED_REFERENCE = re.compile(r"\[E\d+]|\bC\d+\b|\b[0-9a-f]{8}-[0-9a-f-]{27}\b", re.I)


class FinalizerEvidenceOutput(BaseModel):
    candidate_index: int
    role: Literal["core", "supporting"]


class FinalizerAspectOutput(BaseModel):
    aspect_index: int
    evidence: list[FinalizerEvidenceOutput] = Field(default_factory=list)


class FinalizerAnswerBlock(BaseModel):
    text: str
    evidence_indexes: list[int] = Field(default_factory=list)
    kind: Literal["supported", "insufficient"]


class FinalizerOutput(BaseModel):
    aspects: list[FinalizerAspectOutput]
    answer_blocks: list[FinalizerAnswerBlock]


@dataclass(slots=True)
class FinalizerResult:
    answer: str | None
    evidence_pack: EvidencePack
    degraded: bool = False
    error: str | None = None


class ResearchFinalizerService:
    def __init__(
        self,
        provider: LLMProvider,
        evidence_service: ResearchEvidenceService,
        *,
        model: str,
        strong_model_fallback: bool = False,
    ) -> None:
        self.provider = provider
        self.evidence_service = evidence_service
        self.model = model
        self.strong_model_fallback = strong_model_fallback

    async def finalize(
        self,
        *,
        state: ResearchState,
        candidates: list[ResearchCandidate],
        fallback_pack: EvidencePack,
        timeout_seconds: float,
    ) -> FinalizerResult:
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
                    system_prompt=RESEARCH_FINALIZER_PROMPT,
                    user_payload=payload,
                    temperature=0.0,
                    timeout_seconds=timeout_seconds,
                    log_payload=False,
                    max_tokens=FINALIZER_MAX_TOKENS,
                    enable_thinking=False,
                ),
                timeout=timeout_seconds,
            )
            input_tokens = int(result.metadata.input_tokens or 0)
            output_tokens = int(result.metadata.output_tokens or 0)
            state.record_usage(input_tokens=input_tokens, output_tokens=output_tokens)
            selections, blocks = self._validate_output(
                result.output, aspect_count=len(state.aspects), candidate_count=len(candidates)
            )
            validated = await self.evidence_service.build_pack(
                tenant_id=state.tenant_id,
                index_versions=state.index_versions,
                aspects=state.aspects,
                candidates=candidates,
                selections=selections,
            )
            answer = self._render_answer(blocks, validated.evidence_ids)
            state.stages.append(
                ResearchStageUsage(
                    stage="finalizer",
                    role=LLMRole.STRONG,
                    model=self.model,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
            )
            return FinalizerResult(answer=answer, evidence_pack=validated.pack)
        except Exception as exception:
            error = safe_stage_error("finalizer", exception)
            state.degraded = True
            state.errors.append(error)
            state.stages.append(
                ResearchStageUsage(
                    stage="finalizer",
                    role=LLMRole.STRONG,
                    model=self.model,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    degraded=True,
                    error=error,
                )
            )
            return FinalizerResult(
                answer=None, evidence_pack=fallback_pack, degraded=True, error=error
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
                    "rounds": candidate.rounds,
                    "source": candidate.source,
                }
                for index, candidate in enumerate(candidates)
            ],
        }

    @staticmethod
    def _validate_output(
        payload: dict[str, Any], *, aspect_count: int, candidate_count: int
    ) -> tuple[list[list[dict[str, Any]]], list[FinalizerAnswerBlock]]:
        try:
            output = FinalizerOutput.model_validate(payload)
        except ValidationError as exception:
            raise ResearchValidationError("invalid finalizer output") from exception
        if len(output.aspects) != aspect_count or len(output.answer_blocks) > aspect_count + 2:
            raise ResearchValidationError("finalizer output size mismatch")
        selections: list[list[dict[str, Any]] | None] = [None] * aspect_count
        selected: set[int] = set()
        for aspect in output.aspects:
            if (
                isinstance(aspect.aspect_index, bool)
                or aspect.aspect_index < 0
                or aspect.aspect_index >= aspect_count
                or selections[aspect.aspect_index] is not None
            ):
                raise ResearchValidationError("invalid finalizer aspect index")
            seen: set[int] = set()
            references: list[dict[str, Any]] = []
            for reference in aspect.evidence:
                index = reference.candidate_index
                if (
                    isinstance(index, bool)
                    or index < 0
                    or index >= candidate_count
                    or index in seen
                ):
                    raise ResearchValidationError("invalid finalizer evidence index")
                seen.add(index)
                selected.add(index)
                references.append(
                    {"candidate_index": index, "role": reference.role}
                )
            selections[aspect.aspect_index] = references
        if any(item is None for item in selections):
            raise ResearchValidationError("finalizer omitted an aspect")
        total_chars = 0
        for block in output.answer_blocks:
            text = block.text.strip()
            if not text or len(text) > 1000 or _UNRESOLVED_REFERENCE.search(text):
                raise ResearchValidationError("invalid finalizer answer text")
            total_chars += len(text)
            indexes = block.evidence_indexes
            if len(indexes) != len(set(indexes)) or any(
                isinstance(index, bool) or index < 0 or index >= candidate_count
                for index in indexes
            ):
                raise ResearchValidationError("invalid answer evidence index")
            if block.kind == "supported" and (
                not indexes or any(index not in selected for index in indexes)
            ):
                raise ResearchValidationError("supported answer lacks selected evidence")
            if block.kind == "insufficient" and indexes:
                raise ResearchValidationError("insufficient answer cannot cite evidence")
        if not output.answer_blocks or total_chars > MAX_FINAL_ANSWER_CHARS:
            raise ResearchValidationError("final answer size is invalid")
        return [item or [] for item in selections], output.answer_blocks

    @staticmethod
    def _render_answer(
        blocks: list[FinalizerAnswerBlock], evidence_ids: dict[int, str]
    ) -> str:
        rendered: list[str] = []
        for block in blocks:
            citations = "".join(
                f"[{evidence_ids[index]}]"
                for index in block.evidence_indexes
                if index in evidence_ids
            )
            rendered.append(f"{block.text.strip()}{citations}")
        answer = "\n\n".join(rendered).strip()
        if not answer:
            raise ResearchValidationError("final answer is empty")
        return answer
