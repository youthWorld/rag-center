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
