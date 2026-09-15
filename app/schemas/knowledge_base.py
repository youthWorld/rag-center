from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class KnowledgeBaseCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=5000)
    tenant_id: str = Field(min_length=1, max_length=128)

    @field_validator("name", "tenant_id")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class KnowledgeBaseResponse(BaseModel):
    kb_id: str
    name: str
    tenant_id: str
    created_at: datetime


class KnowledgeBaseTreeDocumentResponse(BaseModel):
    document_id: str
    title: str
    status: Literal["SUCCESS"] = "SUCCESS"
    chunk_count: int = Field(ge=0)
    created_at: datetime


class KnowledgeBaseTreeResponse(BaseModel):
    kb_id: str
    name: str
    description: str | None = None
    created_at: datetime
    documents: list[KnowledgeBaseTreeDocumentResponse] = Field(default_factory=list)


class KnowledgeBaseTenantTreeResponse(BaseModel):
    tenant_id: str
    knowledge_bases: list[KnowledgeBaseTreeResponse] = Field(default_factory=list)
