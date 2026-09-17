from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.config import Settings
from app.core.error_codes import ErrorCode
from app.core.exceptions import AppError
from app.schemas.feedback import FeedbackRequest
from app.services.feedback_service import FeedbackService


class FakeFeedbackClient:
    def __init__(
        self,
        *,
        fail_on_score: bool = False,
        fail_on_flush: bool = False,
        existing_feedback: bool = False,
    ) -> None:
        self.fail_on_score = fail_on_score
        self.fail_on_flush = fail_on_flush
        self.existing_feedback = existing_feedback
        self.score_calls: list[dict] = []
        self.flush_count = 0

    def fetch_trace(self, trace_id: str):
        return SimpleNamespace(
            data=SimpleNamespace(
                id=trace_id,
                scores=(
                    [SimpleNamespace(name="user_feedback")]
                    if self.existing_feedback
                    else []
                ),
            )
        )

    def score(self, **kwargs):
        if self.fail_on_score:
            raise RuntimeError("score write failed")
        self.score_calls.append(kwargs)

    def flush(self) -> None:
        self.flush_count += 1
        if self.fail_on_flush:
            raise RuntimeError("flush failed")


def _service(repository, settings: Settings | None = None) -> FeedbackService:
    return FeedbackService(
        settings=settings or Settings(langfuse_enabled=True),
        retrieval_log_repository=repository,
    )


@pytest.mark.asyncio
async def test_feedback_validates_log_ownership_and_writes_score(monkeypatch) -> None:
    repository = SimpleNamespace(
        get_by_id=AsyncMock(
            return_value=SimpleNamespace(
                tenant_id="tenant-a",
                trace_id="trace-a",
            )
        )
    )
    client = FakeFeedbackClient()
    monkeypatch.setattr(
        "app.services.feedback_service.get_langfuse_client",
        lambda _settings: client,
    )

    result = await _service(repository).submit(
        FeedbackRequest(
            trace_id="trace-a",
            log_id="log-a",
            score=4,
            comment="  good result  ",
        ),
        tenant_id="tenant-a",
    )

    assert result.trace_id == "trace-a"
    assert result.log_id == "log-a"
    assert result.score == 4
    assert result.feedback_id
    assert client.score_calls[0]["name"] == "user_feedback"
    assert client.score_calls[0]["value"] == 4
    assert client.score_calls[0]["trace_id"] == "trace-a"
    assert client.score_calls[0]["comment"] == "good result"
    assert client.flush_count == 1


@pytest.mark.asyncio
async def test_feedback_rejects_duplicate_trace_feedback(monkeypatch) -> None:
    client = FakeFeedbackClient(existing_feedback=True)
    monkeypatch.setattr(
        "app.services.feedback_service.get_langfuse_client",
        lambda _settings: client,
    )

    with pytest.raises(AppError) as exception_info:
        await _service(SimpleNamespace()).submit(
            FeedbackRequest(trace_id="trace-a", score=4),
            tenant_id="tenant-a",
        )

    assert exception_info.value.code == ErrorCode.FEEDBACK_ALREADY_SUBMITTED.code
    assert client.score_calls == []


@pytest.mark.asyncio
async def test_feedback_rejects_cross_tenant_or_mismatched_trace(monkeypatch) -> None:
    repository = SimpleNamespace(
        get_by_id=AsyncMock(
            return_value=SimpleNamespace(
                tenant_id="tenant-a",
                trace_id="trace-a",
            )
        )
    )
    monkeypatch.setattr(
        "app.services.feedback_service.get_langfuse_client",
        lambda _settings: FakeFeedbackClient(),
    )

    with pytest.raises(AppError) as exception_info:
        await _service(repository).submit(
            FeedbackRequest(trace_id="trace-b", log_id="log-a", score=3),
            tenant_id="tenant-b",
        )

    assert exception_info.value.code == ErrorCode.FEEDBACK_LOG_MISMATCH.code


@pytest.mark.asyncio
async def test_feedback_returns_20020_when_langfuse_is_disabled(monkeypatch) -> None:
    repository = SimpleNamespace()
    monkeypatch.setattr(
        "app.services.feedback_service.get_langfuse_client",
        lambda _settings: None,
    )

    with pytest.raises(AppError) as exception_info:
        await _service(repository, Settings(langfuse_enabled=False)).submit(
            FeedbackRequest(trace_id="trace-a", score=2),
            tenant_id="tenant-a",
        )

    assert exception_info.value.code == ErrorCode.FEEDBACK_UNAVAILABLE.code


@pytest.mark.asyncio
async def test_feedback_returns_20020_when_langfuse_write_fails(monkeypatch) -> None:
    repository = SimpleNamespace()
    monkeypatch.setattr(
        "app.services.feedback_service.get_langfuse_client",
        lambda _settings: FakeFeedbackClient(fail_on_flush=True),
    )

    with pytest.raises(AppError) as exception_info:
        await _service(repository).submit(
            FeedbackRequest(trace_id="trace-a", score=2),
            tenant_id="tenant-a",
        )

    assert exception_info.value.code == ErrorCode.FEEDBACK_UNAVAILABLE.code


@pytest.mark.asyncio
async def test_feedback_service_defensively_rejects_out_of_range_score(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.feedback_service.get_langfuse_client",
        lambda _settings: FakeFeedbackClient(),
    )

    with pytest.raises(AppError) as exception_info:
        await _service(SimpleNamespace()).submit(
            SimpleNamespace(trace_id="trace-a", log_id=None, score=6, comment=None),
            tenant_id="tenant-a",
        )

    assert exception_info.value.code == ErrorCode.FEEDBACK_SCORE_INVALID.code
