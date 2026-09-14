from typing import Any

from pydantic import BaseModel, Field, field_validator


class RagRetrieveRequest(BaseModel):
    tenant_id: str = Field(min_length=1, max_length=128)
    kb_id: str = Field(min_length=1, max_length=36)
    user_id: str = Field(min_length=1, max_length=128)
    query: str = Field(min_length=1)

    @field_validator("tenant_id", "kb_id", "user_id", "query")
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


class RagRetrieveResponse(BaseModel):
    query: str
    kb_id: str
    retrieved_chunks: list[RetrievedChunk]
    metadata: dict[str, Any]
