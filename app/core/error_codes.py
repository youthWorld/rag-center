from __future__ import annotations

from enum import Enum


class ErrorCode(Enum):
    """Public error codes shared by the API and its clients."""

    SUCCESS = (0, "success")

    # Common client and authentication errors.
    PARAM_ERROR = (10001, "invalid request parameters")
    UNAUTHORIZED = (20010, "authentication required")
    FORBIDDEN = (10003, "permission denied")
    NOT_FOUND = (10004, "resource not found")
    METHOD_ERROR = (10005, "method not allowed")

    # HTTP and upstream API errors.
    API_REQUEST_ERROR = (20001, "request failed")
    API_TIMEOUT = (20002, "request timed out")
    API_RATE_LIMIT = (20003, "request rate limit exceeded")
    REQUEST_VALIDATION_ERROR = (20004, "request validation failed")

    # Database and storage errors.
    DB_ERROR = (30001, "database operation failed")
    DATA_DUPLICATE = (30002, "data already exists")
    STORAGE_NOT_FOUND = (30003, "stored resource not found")

    # Large language model errors.
    LLM_ERROR = (40000, "large language model call failed")
    LLM_TIMEOUT = (40001, "large language model response timed out")
    LLM_NO_RESPONSE = (40002, "large language model returned no valid response")
    LLM_CONTENT_VIOLATION = (40003, "content was rejected")
    LLM_TOKEN_LIMIT = (40004, "context length exceeded")
    LLM_RATE_LIMIT = (40005, "large language model rate limit exceeded")
    LLM_MODEL_ERROR = (40006, "model is unavailable")

    # Internal service errors.
    SERVER_ERROR = (50000, "internal server error")
    SYSTEM_BUSY = (50001, "system is busy, please try again later")
    DOCUMENT_INDEXING_ERROR = (50002, "document indexing failed")
    CONFIGURATION_ERROR = (50003, "service configuration error")

    def __init__(self, code: int, message: str) -> None:
        self.code = code
        self.message = message

    @classmethod
    def from_code(cls, code: int) -> ErrorCode | None:
        for item in cls:
            if item.code == code:
                return item
        return None

    def __int__(self) -> int:
        return self.code

    def __str__(self) -> str:
        return str(self.code)
