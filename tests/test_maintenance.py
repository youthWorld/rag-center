import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from app.api.dependencies import get_knowledge_base_service
from app.core.config import Settings
from app.core.error_codes import ErrorCode
from app.core.exceptions import AppError, LLMServiceError, raise_app_error
from app.core.logging import (
    SizeAndTimeRotatingFileHandler,
    configure_logging,
    format_log_value,
    log_llm_call,
    request_context,
    shutdown_logging,
)
from app.main import app
from app.providers.embedding.openai_compatible import OpenAICompatibleEmbeddingProvider


def test_error_code_and_exception_contract() -> None:
    assert ErrorCode.SUCCESS.value == (0, "success")
    assert 40000 <= ErrorCode.LLM_ERROR.code < 50000
    assert ErrorCode.from_code(50002) is ErrorCode.DOCUMENT_INDEXING_ERROR

    error = AppError(code=ErrorCode.PARAM_ERROR, data={"field": "name"})

    assert error.to_response() == {
        "code": 10001,
        "msg": "invalid request parameters",
        "data": {"field": "name"},
    }

    with pytest.raises(AppError) as raised:
        raise_app_error(ErrorCode.NOT_FOUND, context={"resource": "kb"})

    assert raised.value.code == ErrorCode.NOT_FOUND.code
    assert raised.value.context == {"resource": "kb"}


def test_log_payloads_are_bounded_and_redacted() -> None:
    rendered = format_log_value(
        {"password": "secret", "token": "abc", "content": "x" * 50},
        max_length=30,
    )

    assert "secret" not in rendered
    assert "abc" not in rendered
    assert rendered.endswith("...")


def test_size_and_time_rotating_handler_rotates_by_size(tmp_path) -> None:
    log_path = tmp_path / "app.log"
    handler = SizeAndTimeRotatingFileHandler(
        log_path,
        max_bytes=100,
        backup_count=2,
        when="S",
        interval=3600,
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger("tests.rotating")
    logger.handlers.clear()
    logger.propagate = False
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    for index in range(10):
        logger.info("message-%s-%s", index, "x" * 30)

    handler.flush()
    handler.close()
    logger.removeHandler(handler)

    assert log_path.exists()
    assert (tmp_path / "app.log.1").exists()


def test_configured_logs_keep_normal_and_error_files_separate(tmp_path) -> None:
    original_settings = Settings()
    test_settings = Settings(
        log_dir=str(tmp_path),
        log_max_bytes=1024 * 1024,
        log_backup_count=2,
        log_rotation_when="S",
        log_rotation_interval=3600,
        log_console_color=False,
    )
    configure_logging(test_settings)
    logger = logging.getLogger("tests.files")

    try:
        with request_context("request-file-test"):
            logger.info("normal-event")
            logger.error("error-event")
        shutdown_logging()

        normal_content = (tmp_path / "app.log").read_text(encoding="utf-8")
        error_content = (tmp_path / "error.log").read_text(encoding="utf-8")
    finally:
        configure_logging(original_settings)

    assert "normal-event" in normal_content
    assert "error-event" not in normal_content
    assert "error-event" in error_content
    assert "normal-event" not in error_content
    assert "request-file-test" in normal_content
    assert "request-file-test" in error_content


@pytest.mark.asyncio
async def test_llm_logging_covers_request_and_response(caplog) -> None:
    caplog.set_level(logging.INFO)

    async def operation() -> dict[str, str]:
        return {"answer": "ok"}

    result = await log_llm_call(
        operation,
        model="test-model",
        prompt="hello",
        logger=logging.getLogger("tests.llm"),
    )

    messages = [record.getMessage() for record in caplog.records]
    assert result == {"answer": "ok"}
    assert any("LLM_REQUEST" in message and "test-model" in message for message in messages)
    assert any("LLM_RESPONSE" in message and "answer" in message for message in messages)


@pytest.mark.asyncio
async def test_validation_errors_use_standard_response_and_request_id() -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/knowledge-bases/create",
            headers={"X-Request-ID": "request-test"},
            json={"tenant_id": "tenant-test"},
        )

    assert response.status_code == 400
    assert response.headers["X-Request-ID"] == "request-test"
    assert response.json()["code"] == ErrorCode.REQUEST_VALIDATION_ERROR.code
    assert response.json()["data"]["errors"]


@pytest.mark.asyncio
async def test_unknown_route_uses_standard_http_error() -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/missing")

    assert response.status_code == 404
    assert response.json() == {
        "code": ErrorCode.NOT_FOUND.code,
        "msg": ErrorCode.NOT_FOUND.message,
        "data": None,
    }


@pytest.mark.asyncio
async def test_unexpected_errors_return_safe_response() -> None:
    class FailingKnowledgeBaseService:
        async def create(self, _request):
            raise RuntimeError("database password=secret")

    app.dependency_overrides[get_knowledge_base_service] = lambda: FailingKnowledgeBaseService()
    try:
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/knowledge-bases/create",
                json={
                    "name": "Test KB",
                    "description": "desc",
                    "tenant_id": "tenant-test",
                },
            )
    finally:
        app.dependency_overrides.pop(get_knowledge_base_service, None)

    assert response.status_code == 500
    assert response.json() == {
        "code": ErrorCode.SERVER_ERROR.code,
        "msg": ErrorCode.SERVER_ERROR.message,
        "data": None,
    }


@pytest.mark.asyncio
async def test_embedding_provider_maps_timeout_to_llm_error() -> None:
    provider = OpenAICompatibleEmbeddingProvider(
        Settings(model_api_key="test-key", embedding_dimensions=3)
    )
    create = AsyncMock(side_effect=TimeoutError("upstream timeout"))
    provider._client = SimpleNamespace(embeddings=SimpleNamespace(create=create))

    with pytest.raises(LLMServiceError) as raised:
        await provider.embed_query("question")

    assert raised.value.code == ErrorCode.LLM_TIMEOUT.code
