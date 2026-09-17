from pydantic import BaseModel, Field, field_validator


class FeedbackRequest(BaseModel):
    trace_id: str = Field(min_length=1, max_length=64)
    log_id: str | None = Field(default=None, min_length=1, max_length=36)
    score: int = Field(ge=1, le=5)
    comment: str | None = Field(default=None, max_length=2000)

    @field_validator("trace_id")
    @classmethod
    def normalize_trace_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be blank")
        return normalized

    @field_validator("log_id", "comment")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


class FeedbackData(BaseModel):
    feedback_id: str
    trace_id: str
    log_id: str | None
    score: int
