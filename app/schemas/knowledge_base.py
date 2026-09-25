from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class KnowledgeBaseCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=5000)

    @field_validator("name")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class KnowledgeBaseUpdateRequest(BaseModel):
    name: str | None = Field(default=None, max_length=255)
    description: str | None = Field(default=None, max_length=5000)
    settings: Any = None


class KnowledgeBaseResponse(BaseModel):
    kb_id: str
    name: str
    tenant_id: str
    created_at: datetime
    active_index_version: str = "v1"


class KnowledgeBaseDetailResponse(BaseModel):
    kb_id: str
    name: str
    description: str | None = None
    settings: dict[str, Any]
    document_count: int = Field(ge=0)
    created_at: datetime
    updated_at: datetime
    active_index_version: str = "v1"


class KnowledgeBaseDeleteResponse(BaseModel):
    kb_id: str


class KnowledgeBaseTreeDocumentResponse(BaseModel):
    document_id: str
    title: str
    status: Literal["SUCCESS", "FAILED", "PROCESSING"] = "SUCCESS"
    error_message: str | None = None
    chunk_count: int = Field(ge=0)
    created_at: datetime


class KnowledgeBaseTreeResponse(BaseModel):
    kb_id: str
    name: str
    description: str | None = None
    created_at: datetime
    active_index_version: str = "v1"
    documents: list[KnowledgeBaseTreeDocumentResponse] = Field(default_factory=list)


class KnowledgeBaseTenantTreeResponse(BaseModel):
    tenant_id: str
    knowledge_bases: list[KnowledgeBaseTreeResponse] = Field(default_factory=list)
