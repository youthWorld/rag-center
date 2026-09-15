from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.schemas.hybrid_search import RetrievalOptions, RetrievalSource
from app.schemas.rerank import RerankOptions


class RagRetrieveRequest(BaseModel):
    kb_id: str = Field(min_length=1, max_length=36)
    user_id: str = Field(min_length=1, max_length=128)
    query: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1)
    retrieval_options: RetrievalOptions | None = None
    rerank_options: RerankOptions | None = None

    @field_validator("kb_id", "user_id", "query")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class RetrievedChunk(BaseModel):
    document_id: str
    chunk_id: str
    title: str
    content: str
    score: float
    vector_score: float | None = None
    bm25_score: float | None = None
    vector_rank: int | None = Field(default=None, ge=1)
    bm25_rank: int | None = Field(default=None, ge=1)
    retrieval_source: RetrievalSource = "vector"
    rerank_score: float | None = None


class RagRetrieveResponse(BaseModel):
    query: str
    kb_id: str
    retrieved_chunks: list[RetrievedChunk]
    metadata: dict[str, Any]
