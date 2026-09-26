from __future__ import annotations

import atexit
import inspect
import json
import logging
import logging.handlers
import queue
import sys
import threading
import time
from collections.abc import Awaitable, Callable, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Any, ParamSpec, TypeVar
from uuid import uuid4

from fastapi import Request
from fastapi.responses import Response

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")

REQUEST_ID_HEADER = "X-Request-ID"
_request_id_context: ContextVar[str | None] = ContextVar("request_id", default=None)
_research_retrieval_context: ContextVar[bool] = ContextVar(
    "research_retrieval", default=False
)
_configuration_lock = threading.Lock()
_queue_listener: logging.handlers.QueueListener | None = None
_queue_handler: logging.handlers.QueueHandler | None = None
_atexit_registered = False

_SENSITIVE_KEYS = {
    "api_key",
    "authorization",
    "cookie",
    "password",
    "secret",
    "token",
}


def get_logger(name: str | None = None) -> logging.Logger:
    return logging.getLogger(name or "app")


def get_request_id() -> str | None:
    return _request_id_context.get()


def new_request_id() -> str:
    return uuid4().hex


@contextmanager
def request_context(request_id: str | None = None):
    token = _request_id_context.set(request_id or new_request_id())
    try:
        yield _request_id_context.get()
    finally:
        _request_id_context.reset(token)


@contextmanager
def research_retrieval_log_scope():
    """Hide subqueries from ordinary retrieval logs inside Research tasks."""
    token = _research_retrieval_context.set(True)
    try:
        yield
    finally:
        _research_retrieval_context.reset(token)


class ResearchRetrievalLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if _research_retrieval_context.get() and record.name == "app.services.rag_service":
            event = str(record.msg).split(" |", 1)[0]
            record.msg = f"{event} | research retrieval details redacted"
            record.args = ()
            record.exc_info = None
        return True


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not getattr(record, "request_id", None):
            record.request_id = get_request_id() or "-"
        return True


class MaxLevelFilter(logging.Filter):
    def __init__(self, max_level: int) -> None:
        super().__init__()
        self.max_level = max_level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno <= self.max_level


class ColorFormatter(logging.Formatter):
    _COLORS = {
        logging.DEBUG: "\033[36m",
        logging.INFO: "\033[32m",
        logging.WARNING: "\033[33m",
        logging.ERROR: "\033[31m",
        logging.CRITICAL: "\033[35m",
    }
    _RESET = "\033[0m"

    def __init__(self, *, use_color: bool = True) -> None:
        super().__init__(
            fmt=(
                "%(asctime)s | %(levelname)-8s | %(module)s:%(funcName)s:%(lineno)d | "
                "request_id=%(request_id)s | %(message)s"
            ),
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        if not self.use_color:
            return message
        return f"{self._COLORS.get(record.levelno, '')}{message}{self._RESET}"


class SizeAndTimeRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """Rotate a file when it reaches a size limit or the configured time boundary."""

    _UNIT_SECONDS = {"S": 1, "M": 60, "H": 60 * 60, "D": 24 * 60 * 60}

    def __init__(
        self,
        filename: str | Path,
        *,
        max_bytes: int,
        backup_count: int,
        when: str = "midnight",
        interval: int = 1,
        encoding: str = "utf-8",
    ) -> None:
        self.when = when.upper()
        self.interval = interval
        self._rotation_seconds = self._get_rotation_seconds(self.when, interval)
        super().__init__(
            filename,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding=encoding,
            delay=True,
        )
        self.rollover_at = self._next_rollover(time.time())

    @classmethod
    def _get_rotation_seconds(cls, when: str, interval: int) -> int:
        if interval < 1:
            raise ValueError("log rotation interval must be at least 1")
        if when == "MIDNIGHT":
            return 24 * 60 * 60 * interval
        if when not in cls._UNIT_SECONDS:
            raise ValueError("log rotation when must be one of S, M, H, D, or midnight")
        return cls._UNIT_SECONDS[when] * interval

    def _next_rollover(self, current_time: float) -> float:
        if self.when == "MIDNIGHT":
            now = datetime.now()
            next_midnight = (now + timedelta(days=self.interval)).replace(
                hour=0,
                minute=0,
                second=0,
                microsecond=0,
            )
            return next_midnight.timestamp()
        return current_time + self._rotation_seconds

    def shouldRollover(self, record: logging.LogRecord) -> int:  # noqa: N802
        if self.stream is None:
            self.stream = self._open()
        if self.maxBytes > 0:
            message = f"{self.format(record)}\n"
            encoded_message = message.encode(self.encoding or "utf-8", errors="replace")
            if self.stream.tell() + len(encoded_message) >= self.maxBytes:
                return 1
        if time.time() >= self.rollover_at:
            return 1
        return 0

    def doRollover(self) -> None:  # noqa: N802
        super().doRollover()
        self.rollover_at = self._next_rollover(time.time())


def _redact(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): "***" if str(key).lower() in _SENSITIVE_KEYS else _redact(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_redact(item) for item in value]
    return value


def format_log_value(value: Any, *, max_length: int = 2000) -> str:
    """Serialize structured values safely and keep payload logging bounded."""

    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    value = _redact(value)
    try:
        rendered = (
            value
            if isinstance(value, str)
            else json.dumps(value, ensure_ascii=False, default=str)
        )
    except (TypeError, ValueError):
        rendered = repr(value)
    if len(rendered) <= max_length:
        return rendered
    return f"{rendered[:max_length]}..."


def _level_from_name(level: str) -> int:
    resolved = logging.getLevelName(level.upper())
    return resolved if isinstance(resolved, int) else logging.INFO


def configure_logging(settings: Any | None = None) -> None:
    """Configure one process-wide, queue-backed logging pipeline."""

    global _queue_handler, _queue_listener, _atexit_registered
    if settings is None:
        from app.core.config import settings as default_settings

        settings = default_settings

    with _configuration_lock:
        if _queue_listener is not None:
            _queue_listener.stop()
            for listener_handler in _queue_listener.handlers:
                listener_handler.close()
            _queue_listener = None

        root_logger = logging.getLogger()
        if _queue_handler is not None:
            root_logger.removeHandler(_queue_handler)
            _queue_handler.close()
            _queue_handler = None
        for handler in list(root_logger.handlers):
            if getattr(handler, "_rag_center_handler", False):
                root_logger.removeHandler(handler)
                handler.close()

        log_dir = Path(settings.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        level = _level_from_name(settings.log_level)
        formatter = ColorFormatter(use_color=settings.log_console_color)
        plain_formatter = ColorFormatter(use_color=False)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        console_handler.addFilter(RequestIdFilter())

        file_handler = SizeAndTimeRotatingFileHandler(
            log_dir / "app.log",
            max_bytes=settings.log_max_bytes,
            backup_count=settings.log_backup_count,
            when=settings.log_rotation_when,
            interval=settings.log_rotation_interval,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(plain_formatter)
        file_handler.addFilter(RequestIdFilter())
        file_handler.addFilter(MaxLevelFilter(logging.WARNING))

        error_handler = SizeAndTimeRotatingFileHandler(
            log_dir / "error.log",
            max_bytes=settings.log_max_bytes,
            backup_count=settings.log_backup_count,
            when=settings.log_rotation_when,
            interval=settings.log_rotation_interval,
            encoding="utf-8",
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(plain_formatter)
        error_handler.addFilter(RequestIdFilter())

        log_queue: queue.Queue[logging.LogRecord] = queue.Queue(-1)
        queue_handler = logging.handlers.QueueHandler(log_queue)
        queue_handler.setLevel(level)
        queue_handler.addFilter(RequestIdFilter())
        queue_handler.addFilter(ResearchRetrievalLogFilter())
        queue_handler._rag_center_handler = True  # type: ignore[attr-defined]
        _queue_handler = queue_handler

        root_logger.setLevel(level)
        root_logger.addHandler(queue_handler)
        _queue_listener = logging.handlers.QueueListener(
            log_queue,
            console_handler,
            file_handler,
            error_handler,
            respect_handler_level=True,
        )
        _queue_listener.start()

        if not _atexit_registered:
            atexit.register(shutdown_logging)
            _atexit_registered = True


def shutdown_logging() -> None:
    global _queue_handler, _queue_listener
    with _configuration_lock:
        if _queue_listener is not None:
            _queue_listener.stop()
            for listener_handler in _queue_listener.handlers:
                listener_handler.close()
            _queue_listener = None
        root_logger = logging.getLogger()
        if _queue_handler is not None:
            root_logger.removeHandler(_queue_handler)
            _queue_handler.close()
            _queue_handler = None


def _request_details(request: Request) -> dict[str, str]:
    return {
        "method": request.method,
        "url": str(request.url),
        "client": str(request.client.host) if request.client else "-",
    }


def log_exception(
    exception: BaseException,
    *,
    logger: logging.Logger | None = None,
    context: Mapping[str, Any] | None = None,
) -> None:
    """Write an exception with its complete traceback to the error log."""

    active_logger = logger or get_logger(__name__)
    active_logger.error(
        "SYSTEM_ERROR | exception_type=%s | error=%s | context=%s",
        type(exception).__name__,
        str(exception),
        format_log_value(dict(context or {})),
        exc_info=(type(exception), exception, exception.__traceback__),
    )


def _status_code_from_result(result: Any) -> int | None:
    return getattr(result, "status_code", None)


def _find_request(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> Request | None:
    for value in (*args, *kwargs.values()):
        if isinstance(value, Request):
            return value
    return None


def log_api_call(func: Callable[P, R]) -> Callable[P, R]:
    """Log function-level API request and response details."""

    if inspect.iscoroutinefunction(func):

        @wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            started = time.perf_counter()
            request = _find_request(args, kwargs)
            logger = get_logger(func.__module__)
            logger.info(
                "API_REQUEST | function=%s | method=%s | url=%s | args=%s | kwargs=%s",
                func.__qualname__,
                request.method if request else "-",
                str(request.url) if request else "-",
                format_log_value(args),
                format_log_value(kwargs),
            )
            try:
                result = await func(*args, **kwargs)
            except Exception as exception:
                logger.exception(
                    "API_ERROR | function=%s | cost_ms=%.2f | exception_type=%s",
                    func.__qualname__,
                    (time.perf_counter() - started) * 1000,
                    type(exception).__name__,
                )
                raise
            logger.info(
                "API_RESPONSE | function=%s | status_code=%s | cost_ms=%.2f | result=%s",
                func.__qualname__,
                _status_code_from_result(result) or 200,
                (time.perf_counter() - started) * 1000,
                format_log_value(result),
            )
            return result

        return async_wrapper

    @wraps(func)
    def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        started = time.perf_counter()
        request = _find_request(args, kwargs)
        logger = get_logger(func.__module__)
        logger.info(
            "API_REQUEST | function=%s | method=%s | url=%s | args=%s | kwargs=%s",
            func.__qualname__,
            request.method if request else "-",
            str(request.url) if request else "-",
            format_log_value(args),
            format_log_value(kwargs),
        )
        try:
            result = func(*args, **kwargs)
        except Exception as exception:
            logger.exception(
                "API_ERROR | function=%s | cost_ms=%.2f | exception_type=%s",
                func.__qualname__,
                (time.perf_counter() - started) * 1000,
                type(exception).__name__,
            )
            raise
        logger.info(
            "API_RESPONSE | function=%s | status_code=%s | cost_ms=%.2f | result=%s",
            func.__qualname__,
            _status_code_from_result(result) or 200,
            (time.perf_counter() - started) * 1000,
            format_log_value(result),
        )
        return result

    return sync_wrapper


async def log_llm_call(
    operation: Callable[[], Awaitable[T]],
    *,
    model: str,
    prompt: Any,
    logger: logging.Logger | None = None,
    response_formatter: Callable[[T], Any] | None = None,
    max_length: int = 2000,
    safe_error: bool = False,
) -> T:
    """Run an async model operation with request, response, latency, and error logs."""

    active_logger = logger or get_logger("app.llm")
    started = time.perf_counter()
    active_logger.info(
        "LLM_REQUEST | model=%s | prompt=%s",
        model,
        _format_llm_prompt(prompt, max_length=max_length),
    )
    try:
        response = await operation()
    except Exception as exception:
        if safe_error or _research_retrieval_context.get():
            active_logger.error(
                "LLM_ERROR | model=%s | cost_ms=%.2f | exception_type=%s",
                model,
                (time.perf_counter() - started) * 1000,
                type(exception).__name__,
            )
        else:
            active_logger.exception(
                "LLM_ERROR | model=%s | cost_ms=%.2f | exception_type=%s | error=%s | prompt=%s",
                model,
                (time.perf_counter() - started) * 1000,
                type(exception).__name__,
                str(exception),
                _format_llm_prompt(prompt, max_length=max_length),
            )
        raise

    rendered_response = response_formatter(response) if response_formatter else response
    active_logger.info(
        "LLM_RESPONSE | model=%s | cost_ms=%.2f | response=%s",
        model,
        (time.perf_counter() - started) * 1000,
        format_log_value(rendered_response, max_length=max_length),
    )
    return response


def _format_llm_prompt(prompt: Any, *, max_length: int) -> str:
    """Keep model logs useful without copying document contents into log files."""

    del max_length
    if isinstance(prompt, list) and all(isinstance(item, str) for item in prompt):
        return f"<text_batch count={len(prompt)} chars={sum(len(item) for item in prompt)}>"
    if isinstance(prompt, str):
        return f"<text chars={len(prompt)}>"
    if isinstance(prompt, dict):
        return f"<object keys={sorted(str(key) for key in prompt)}>"
    return f"<{type(prompt).__name__}>"


log_api = log_api_call
log_llm_interaction = log_llm_call


async def request_logging_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Capture every HTTP request and response while preserving the response body."""

    request_id = request.headers.get(REQUEST_ID_HEADER) or new_request_id()
    request.state.request_id = request_id
    request_token = _request_id_context.set(request_id)
    logger = get_logger("app.http")
    started = time.perf_counter()
    body = await request.body()
    settings = getattr(request.app.state, "settings", None)
    max_length = getattr(settings, "log_payload_max_length", 2000)
    details = _request_details(request)
    logger.info(
        "API_REQUEST | method=%s | url=%s | body=%s",
        details["method"],
        details["url"],
        (
            f"<research request bytes={len(body)}>"
            if request.url.path == "/api/v1/rag/research"
            else format_log_value(body, max_length=max_length)
        ),
    )

    try:
        response = await call_next(request)
    except Exception as exception:
        log_exception(
            exception,
            logger=logger,
            context={
                **details,
                "cost_ms": round((time.perf_counter() - started) * 1000, 2),
            },
        )
        _request_id_context.reset(request_token)
        raise

    response_body = getattr(response, "body", None)
    if response_body is not None:
        rendered_body = response_body
    else:
        chunks = [chunk async for chunk in response.body_iterator]
        rendered_body = b"".join(chunks)
        response = Response(
            content=rendered_body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
            background=response.background,
        )

    response.headers[REQUEST_ID_HEADER] = request_id
    logger.info(
        "API_RESPONSE | method=%s | url=%s | status_code=%s | cost_ms=%.2f | response=%s",
        details["method"],
        details["url"],
        response.status_code,
        (time.perf_counter() - started) * 1000,
        (
            f"<research response bytes={len(rendered_body)}>"
            if request.url.path == "/api/v1/rag/research"
            else format_log_value(rendered_body, max_length=max_length)
        ),
    )
    _request_id_context.reset(request_token)
    return response
