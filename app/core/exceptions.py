from __future__ import annotations

from typing import Any, NoReturn

from app.core.error_codes import ErrorCode

ErrorCodes = ErrorCode

_DEFAULT_STATUS_CODES = {
    ErrorCode.SUCCESS: 200,
    ErrorCode.PARAM_ERROR: 400,
    ErrorCode.UNAUTHORIZED: 401,
    ErrorCode.FORBIDDEN: 403,
    ErrorCode.NOT_FOUND: 404,
    ErrorCode.METHOD_ERROR: 405,
    ErrorCode.API_REQUEST_ERROR: 502,
    ErrorCode.API_TIMEOUT: 504,
    ErrorCode.API_RATE_LIMIT: 429,
    ErrorCode.REQUEST_VALIDATION_ERROR: 400,
    ErrorCode.FEATURE_NOT_ALLOWED: 403,
    ErrorCode.QUOTA_EXCEEDED: 429,
    ErrorCode.FEEDBACK_UNAVAILABLE: 502,
    ErrorCode.FEEDBACK_LOG_MISMATCH: 400,
    ErrorCode.FEEDBACK_SCORE_INVALID: 400,
    ErrorCode.FEEDBACK_ALREADY_SUBMITTED: 409,
    ErrorCode.DOCUMENT_PARSE_FAILED: 400,
    ErrorCode.DB_ERROR: 500,
    ErrorCode.DATA_DUPLICATE: 409,
    ErrorCode.STORAGE_NOT_FOUND: 404,
    ErrorCode.LLM_ERROR: 502,
    ErrorCode.LLM_TIMEOUT: 504,
    ErrorCode.LLM_NO_RESPONSE: 502,
    ErrorCode.LLM_CONTENT_VIOLATION: 422,
    ErrorCode.LLM_TOKEN_LIMIT: 400,
    ErrorCode.LLM_RATE_LIMIT: 429,
    ErrorCode.LLM_MODEL_ERROR: 502,
    ErrorCode.SERVER_ERROR: 500,
    ErrorCode.SYSTEM_BUSY: 503,
    ErrorCode.DOCUMENT_INDEXING_ERROR: 500,
    ErrorCode.CONFIGURATION_ERROR: 500,
}


def _resolve_error_code(code: int | ErrorCode) -> ErrorCode | None:
    if isinstance(code, ErrorCode):
        return code
    return ErrorCode.from_code(code)


class AppError(Exception):
    """An expected application error with a safe public response and private context."""

    def __init__(
        self,
        message: str | None = None,
        *,
        code: int | ErrorCode = ErrorCode.SERVER_ERROR,
        status_code: int | None = None,
        data: Any = None,
        internal_message: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        self.error_code = _resolve_error_code(code)
        self.code = self.error_code.code if self.error_code else int(code)
        default_message = self.error_code.message if self.error_code else "application error"
        self.message = message or default_message
        self.status_code = status_code or _DEFAULT_STATUS_CODES.get(
            self.error_code,
            500,
        )
        self.data = data
        self.internal_message = internal_message or self.message
        self.context = context or {}
        super().__init__(self.internal_message)

    def to_response(self) -> dict[str, Any]:
        return {"code": self.code, "msg": self.message, "data": self.data}


def raise_app_error(
    code: int | ErrorCode,
    message: str | None = None,
    *,
    status_code: int | None = None,
    data: Any = None,
    internal_message: str | None = None,
    context: dict[str, Any] | None = None,
) -> NoReturn:
    """Raise a standard application error without repeating constructor boilerplate."""

    raise AppError(
        message,
        code=code,
        status_code=status_code,
        data=data,
        internal_message=internal_message,
        context=context,
    )


class KnowledgeBaseNotFoundError(AppError):
    def __init__(
        self,
        kb_id: str | None = None,
        *,
        missing_kb_id: str | None = None,
    ) -> None:
        context: dict[str, Any] = {}
        if kb_id:
            context["kb_id"] = kb_id
        if missing_kb_id:
            context["missing_kb_id"] = missing_kb_id
        super().__init__(
            "knowledge base not found",
            code=ErrorCode.NOT_FOUND,
            context=context or None,
        )


class DocumentNotFoundError(AppError):
    def __init__(self, document_id: str | None = None) -> None:
        super().__init__(
            "document not found",
            code=ErrorCode.NOT_FOUND,
            context={"document_id": document_id} if document_id else None,
        )


class DocumentIndexingError(AppError):
    def __init__(self, document_id: str, message: str) -> None:
        super().__init__(
            "document indexing failed",
            code=ErrorCode.DOCUMENT_INDEXING_ERROR,
            data={"document_id": document_id, "status": 2},
            internal_message=f"document indexing failed: {message}",
            context={"document_id": document_id},
        )


class DocumentParseError(AppError):
    def __init__(self, message: str, *, filename: str | None = None) -> None:
        super().__init__(
            "document parsing failed",
            code=ErrorCode.DOCUMENT_PARSE_FAILED,
            internal_message=message,
            context={"filename": filename} if filename else None,
        )


class LLMServiceError(AppError):
    def __init__(
        self,
        message: str | None = None,
        *,
        code: ErrorCode = ErrorCode.LLM_ERROR,
        internal_message: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message or code.message,
            code=code,
            internal_message=internal_message,
            context=context,
        )


class ServiceConfigurationError(AppError):
    def __init__(self, *, internal_message: str, context: dict[str, Any] | None = None) -> None:
        super().__init__(
            code=ErrorCode.CONFIGURATION_ERROR,
            internal_message=internal_message,
            context=context,
        )


class FeedbackUnavailableError(AppError):
    def __init__(self, *, internal_message: str | None = None) -> None:
        super().__init__(
            code=ErrorCode.FEEDBACK_UNAVAILABLE,
            internal_message=internal_message,
        )


class FeedbackLogMismatchError(AppError):
    def __init__(self) -> None:
        super().__init__(code=ErrorCode.FEEDBACK_LOG_MISMATCH)


class FeedbackScoreInvalidError(AppError):
    def __init__(self) -> None:
        super().__init__(code=ErrorCode.FEEDBACK_SCORE_INVALID)


class FeedbackAlreadySubmittedError(AppError):
    def __init__(self) -> None:
        super().__init__(code=ErrorCode.FEEDBACK_ALREADY_SUBMITTED)


def map_llm_exception(
    exception: Exception,
    *,
    model: str,
    operation: str,
) -> LLMServiceError:
    """Map common provider failures to stable public LLM error codes."""

    message = str(exception).lower()
    exception_name = type(exception).__name__.lower()
    if "timeout" in message or "timeout" in exception_name:
        code = ErrorCode.LLM_TIMEOUT
    elif "rate" in message or "429" in message or "ratelimit" in exception_name:
        code = ErrorCode.LLM_RATE_LIMIT
    elif "context" in message or "token" in message or "length" in message:
        code = ErrorCode.LLM_TOKEN_LIMIT
    elif "content" in message or "moderation" in message or "safety" in message:
        code = ErrorCode.LLM_CONTENT_VIOLATION
    elif "model" in message or "deployment" in message:
        code = ErrorCode.LLM_MODEL_ERROR
    else:
        code = ErrorCode.LLM_ERROR

    return LLMServiceError(
        code=code,
        internal_message=str(exception),
        context={
            "model": model,
            "operation": operation,
            "exception_type": type(exception).__name__,
        },
    )
