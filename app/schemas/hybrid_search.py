from typing import Any, Literal

from pydantic import BaseModel, Field

RetrievalMode = Literal["vector", "bm25", "hybrid"]
RetrievalSource = Literal["vector", "bm25", "hybrid", "graph"]


class RetrievalOptions(BaseModel):
    mode: RetrievalMode | None = None
    vector_top_k: int | None = Field(default=None, ge=1)
    bm25_top_k: int | None = Field(default=None, ge=1)
    rrf_k: int | None = Field(default=None, ge=1)


class HybridSearchChunk(BaseModel):
    document_id: str
    chunk_id: str
    title: str
    content: str
    score: float
    vector_score: float | None = None
    bm25_score: float | None = None
    vector_rank: int | None = Field(default=None, ge=1)
    bm25_rank: int | None = Field(default=None, ge=1)
    retrieval_source: RetrievalSource
    index_version: str | None = None
    retrieval_text: str | None = None
    section_id: str | None = None
    parent_section_id: str | None = None
    order_index: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
