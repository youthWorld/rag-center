from pydantic import BaseModel, Field


class RerankOptions(BaseModel):
    enabled: bool | None = None
    top_n: int | None = Field(default=None, ge=1)


class RerankCandidate(BaseModel):
    chunk_id: str
    document_id: str
    title: str
    content: str
    vector_score: float


class RerankRanking(BaseModel):
    chunk_id: str
    rerank_score: float = Field(ge=0.0, le=1.0)
    reason: str | None = None


class RerankResponse(BaseModel):
    rankings: list[RerankRanking]
