from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.schemas.rag import MULTI_KB_MAX, EvidencePack

ResearchStopReason = Literal[
    "evidence_complete",
    "no_new_evidence",
    "max_rounds",
    "duplicate_query",
    "timeout",
    "budget_exhausted",
    "degraded",
]
ResearchStage = Literal[
    "planning",
    "retrieving",
    "deciding",
    "supplementing",
    "finalizing",
    "completed",
]


def normalize_research_query(value: str) -> str:
    normalized = " ".join(value.strip().lower().split())
    return normalized.rstrip("?？.。")


class ResearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kb_id: str | None = Field(default=None, min_length=1, max_length=36)
    kb_ids: list[str] | None = Field(default=None, min_length=1, max_length=MULTI_KB_MAX)
    user_id: str = Field(min_length=1, max_length=128)
    query: str = Field(min_length=1)

    @field_validator("kb_id", "user_id")
    @classmethod
    def reject_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be blank")
        return normalized

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be blank")
        return normalized

    @field_validator("kb_ids")
    @classmethod
    def normalize_kb_ids(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        normalized: list[str] = []
        for kb_id in value:
            item = kb_id.strip()
            if not item:
                raise ValueError("kb_ids must not contain blank values")
            if len(item) > 36:
                raise ValueError("kb_id must not exceed 36 characters")
            if item not in normalized:
                normalized.append(item)
        return normalized

    @model_validator(mode="after")
    def require_knowledge_base_selector(self) -> Self:
        if self.kb_id is None and not self.kb_ids:
            raise ValueError("either kb_id or kb_ids must be provided")
        return self

    def resolved_kb_ids(self) -> list[str]:
        if self.kb_ids is not None:
            return list(self.kb_ids)
        return [self.kb_id] if self.kb_id is not None else []


class ResearchAspect(BaseModel):
    aspect_id: str
    description: str


class PlannedQuery(BaseModel):
    query_id: str
    query: str
    aspect_ids: list[str]


class ResearchTaskResult(BaseModel):
    query_id: str
    query: str
    aspect_ids: list[str]
    round: int
    success: bool
    latency_ms: int
    chunk_count: int = 0
    new_chunk_count: int = 0
    degraded: bool = False
    error: str | None = None


class ResearchRound(BaseModel):
    round: int
    purpose: Literal["initial", "supplement"]
    tasks: list[ResearchTaskResult] = Field(default_factory=list)
    candidate_count: int = 0
    new_chunk_count: int = 0
    latency_ms: int = 0


class ResearchStageUsage(BaseModel):
    stage: Literal["planner", "decider", "finalizer"]
    role: Literal["fast", "strong"]
    model: str
    latency_ms: int
    input_tokens: int = 0
    output_tokens: int = 0
    degraded: bool = False
    error: str | None = None


class ResearchState(BaseModel):
    research_id: str
    original_query: str
    tenant_id: str
    kb_ids: list[str]
    index_versions: dict[str, str]
    stage: ResearchStage = "planning"
    aspects: list[ResearchAspect] = Field(default_factory=list)
    initial_queries: list[PlannedQuery] = Field(default_factory=list)
    missing_aspect_ids: list[str] = Field(default_factory=list)
    rounds: list[ResearchRound] = Field(default_factory=list)
    visited_queries: list[str] = Field(default_factory=list)
    candidate_chunk_ids: list[str] = Field(default_factory=list)
    round_count: int = 0
    retrieval_task_count: int = 0
    llm_call_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    elapsed_ms: int = 0
    stop_reason: ResearchStopReason | None = None
    degraded: bool = False
    errors: list[str] = Field(default_factory=list)
    stages: list[ResearchStageUsage] = Field(default_factory=list)

    def has_visited(self, query: str) -> bool:
        return normalize_research_query(query) in self.visited_queries

    def mark_visited(self, query: str) -> bool:
        normalized = normalize_research_query(query)
        if normalized in self.visited_queries:
            return False
        self.visited_queries.append(normalized)
        return True

    def record_llm_attempt(self) -> None:
        self.llm_call_count += 1

    def record_usage(self, *, input_tokens: int = 0, output_tokens: int = 0) -> None:
        self.input_tokens += max(0, input_tokens)
        self.output_tokens += max(0, output_tokens)


class ResearchCandidate(BaseModel):
    candidate_id: str = ""
    chunk_id: str
    kb_id: str
    document_id: str
    index_version: str
    title: str
    heading_path: str | None = None
    content: str
    retrieval_text: str | None = None
    source: str
    aspect_ids: list[str] = Field(default_factory=list)
    query_ids: list[str] = Field(default_factory=list)
    rounds: list[int] = Field(default_factory=list)
    best_rank: int | None = None
    fusion_score: float = 0.0
    context_relation: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResearchPlanData(BaseModel):
    aspects: list[ResearchAspect]
    initial_queries: list[PlannedQuery]
    degraded: bool = False
    error: str | None = None


class ResearchMetadata(BaseModel):
    research_id: str
    log_id: str
    trace_id: str | None = None
    profile: str = "research_fixed"
    tenant_plan: str
    index_versions: dict[str, str]
    round_count: int
    retrieval_task_count: int
    llm_call_count: int
    input_tokens: int
    output_tokens: int
    latency_ms: int
    stop_reason: ResearchStopReason
    first_round_missing_aspect_ids: list[str]
    degraded: bool = False
    errors: list[str] = Field(default_factory=list)
    stages: list[ResearchStageUsage] = Field(default_factory=list)
    planner: dict[str, Any] = Field(default_factory=dict)
    decider: dict[str, Any] = Field(default_factory=dict)
    finalizer: dict[str, Any] = Field(default_factory=dict)


class ResearchData(BaseModel):
    query: str
    kb_id: str
    kb_ids: list[str]
    answer: str | None
    evidence_pack: EvidencePack
    plan: ResearchPlanData
    rounds: list[ResearchRound]
    metadata: ResearchMetadata
