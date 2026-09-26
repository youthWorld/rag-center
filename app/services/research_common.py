from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from app.core.error_codes import ErrorCode
from app.core.exceptions import AppError
from app.services.research_limits import MAX_LLM_INPUT_CHARS, MAX_QUERY_CHARS

_UNSAFE_QUERY = re.compile(
    r"(?:https?://|\b(?:select|insert|update|delete|drop|alter)\b|"
    r"(?:^|\s)(?:curl|wget|powershell|cmd|bash|sh)(?:\s|$)|"
    r"(?:调用|使用).{0,8}(?:浏览器|工具|shell|终端))",
    re.IGNORECASE,
)


class ResearchValidationError(ValueError):
    pass


def validate_search_query(value: Any) -> str:
    if not isinstance(value, str):
        raise ResearchValidationError("query must be a string")
    query = value.strip()
    if not query or len(query) > MAX_QUERY_CHARS:
        raise ResearchValidationError("query length is invalid")
    if _UNSAFE_QUERY.search(query):
        raise ResearchValidationError("query contains an unsupported command or tool")
    return query


def payload_chars(payload: dict[str, Any]) -> int:
    return len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


def ensure_payload_budget(payload: dict[str, Any]) -> None:
    if payload_chars(payload) > MAX_LLM_INPUT_CHARS:
        raise ResearchValidationError("research model input exceeds the character budget")


def safe_stage_error(stage: str, exception: BaseException) -> str:
    if isinstance(exception, (TimeoutError, asyncio.TimeoutError)) or (
        isinstance(exception, AppError)
        and exception.error_code in {ErrorCode.API_TIMEOUT, ErrorCode.LLM_TIMEOUT}
    ):
        return f"{stage} timeout"
    if "budget" in str(exception).lower():
        return f"{stage} budget exhausted"
    return f"{stage} failed ({type(exception).__name__})"
