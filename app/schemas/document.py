from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class DocumentUploadRequest(BaseModel):
    kb_id: str = Field(min_length=1, max_length=36)
    title: str = Field(min_length=1, max_length=255)
    content: str = Field(min_length=1)

    @field_validator("kb_id", "title", "content")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class DocumentUploadResponse(BaseModel):
    document_id: str
    kb_id: str
    status: int
    chunk_count: int


class DocumentDetailResponse(BaseModel):
    document_id: str
    kb_id: str
    title: str
    status: int
    error_message: str | None = None
    chunk_count: int = Field(ge=0)
    created_at: datetime
    updated_at: datetime


class DocumentDeleteResponse(BaseModel):
    document_id: str
