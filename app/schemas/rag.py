from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator
from typing_extensions import TypedDict

from app.schemas.hybrid_search import RetrievalOptions, RetrievalSource
from app.schemas.rerank import RerankOptions

QueryRewriteStrategy = Literal["noop", "rewrite"]
RetrieveProfile = Literal["speed", "balanced", "quality", "custom"]
MULTI_KB_MAX = 5


class QueryOptions(BaseModel):
    enabled: bool | None = None
    strategy: QueryRewriteStrategy | None = None


class RagRetrieveRequest(BaseModel):
    kb_id: str | None = Field(default=None, min_length=1, max_length=36)
    kb_ids: list[str] | None = Field(
        default=None,
        min_length=1,
        max_length=MULTI_KB_MAX,
    )
    user_id: str = Field(min_length=1, max_length=128)
    query: str = Field(min_length=1)
    profile: RetrieveProfile | None = None
    top_k: int | None = Field(default=None, ge=1)
    retrieval_options: RetrievalOptions | None = None
    rerank_options: RerankOptions | None = None
    query_options: QueryOptions | None = None

    @field_validator("kb_id", "user_id", "query")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("kb_ids")
    @classmethod
    def normalize_kb_ids(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        normalized: list[str] = []
        for kb_id in value:
            kb_id = kb_id.strip()
            if not kb_id:
                raise ValueError("kb_ids must not contain blank values")
            if len(kb_id) > 36:
                raise ValueError("kb_id must not exceed 36 characters")
            normalized.append(kb_id)
        return normalized

    @model_validator(mode="after")
    def require_knowledge_base_selector(self) -> "RagRetrieveRequest":
        if self.kb_id is None and not self.kb_ids:
            raise ValueError("either kb_id or kb_ids must be provided")
        return self


class RetrievedChunk(BaseModel):
    document_id: str
    chunk_id: str
    kb_id: str | None = Field(default=None, min_length=1, max_length=36)
    kb_name: str | None = None
    title: str
    content: str
    score: float
    vector_score: float | None = None
    bm25_score: float | None = None
    vector_rank: int | None = Field(default=None, ge=1)
    bm25_rank: int | None = Field(default=None, ge=1)
    retrieval_source: RetrievalSource = "vector"
    rerank_score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrieveMetadata(TypedDict, total=False):
    log_id: str
    trace_id: str | None
    top_k: int
    latency_ms: int
    vector_store: str
    query_processing: dict[str, Any] | None
    retrieval: dict[str, Any]
    rerank: dict[str, Any]
    tenant_policy: dict[str, Any]


class RagRetrieveResponse(BaseModel):
    query: str
    kb_id: str
    kb_ids: list[str] = Field(default_factory=list)
    retrieved_chunks: list[RetrievedChunk]
    metadata: RetrieveMetadata

    @model_validator(mode="after")
    def fill_legacy_kb_ids(self) -> "RagRetrieveResponse":
        if not self.kb_ids:
            self.kb_ids = [self.kb_id]
        return self
