from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

IndexVersionStatusValue = Literal["building", "ready", "active", "failed", "archived"]


class IndexVersionResponse(BaseModel):
    version: str
    status: IndexVersionStatusValue
    document_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    error_message: str | None = None
    created_at: datetime
    completed_at: datetime | None = None
    activated_at: datetime | None = None


class IndexVersionRebuildResponse(BaseModel):
    kb_id: str
    version: str
    status: IndexVersionStatusValue


class IndexVersionListResponse(BaseModel):
    kb_id: str
    active_index_version: str
    versions: list[IndexVersionResponse]
