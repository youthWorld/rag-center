from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.core.error_codes import ErrorCode
from app.core.exceptions import AppError
from app.schemas.feedback import FeedbackRequest
from app.services.feedback_service import FeedbackService


class FakeFeedbackClient:
    def __init__(
        self,
        *,
        tenant_id: str | None = "tenant-a",
        log_id: str | None = "log-a",
        scores: list[object] | None = None,
        fail_on_score: bool = False,
        fail_on_flush: bool = False,
    ) -> None:
        self.tenant_id = tenant_id
        self.log_id = log_id
        self.scores = scores or []
        self.fail_on_score = fail_on_score
        self.fail_on_flush = fail_on_flush
        self.score_calls: list[dict] = []
        self.flush_count = 0

    def fetch_trace(self, trace_id: str):
        return SimpleNamespace(
            data=SimpleNamespace(
                id=trace_id,
                metadata=(
                    {"tenant_id": self.tenant_id, "log_id": self.log_id}
                    if self.tenant_id is not None or self.log_id is not None
                    else None
                ),
                scores=self.scores,
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


def _service(settings: Settings | None = None) -> FeedbackService:
    return FeedbackService(settings=settings or Settings(langfuse_enabled=True))


def _request(*, score: int = 4, log_id: str = "log-a", comment: str | None = None):
    return FeedbackRequest(
        trace_id="trace-a",
        log_id=log_id,
        score=score,
        comment=comment,
    )


@pytest.mark.asyncio
async def test_feedback_validates_trace_metadata_and_writes_deterministic_score(
    monkeypatch,
) -> None:
    client = FakeFeedbackClient()
    monkeypatch.setattr(
        "app.services.feedback_service.get_langfuse_client",
        lambda _settings: client,
    )

    result = await _service().submit(
        _request(comment="  good result  "),
        tenant_id="tenant-a",
    )

    assert result.trace_id == "trace-a"
    assert result.log_id == "log-a"
    assert result.score == 4
    assert result.feedback_id
    assert client.score_calls[0]["id"] == result.feedback_id
    assert client.score_calls[0]["name"] == "user_feedback"
    assert client.score_calls[0]["value"] == 4
    assert client.score_calls[0]["trace_id"] == "trace-a"
    assert client.score_calls[0]["comment"] == "good result"
    assert client.flush_count == 1


@pytest.mark.asyncio
async def test_feedback_reuses_deterministic_id_for_repeated_updates(monkeypatch) -> None:
    client = FakeFeedbackClient()
    monkeypatch.setattr(
        "app.services.feedback_service.get_langfuse_client",
        lambda _settings: client,
    )

    first = await _service().submit(_request(score=4), tenant_id="tenant-a")
    second = await _service().submit(_request(score=2), tenant_id="tenant-a")

    assert first.feedback_id == second.feedback_id
    assert [call["id"] for call in client.score_calls] == [first.feedback_id, first.feedback_id]
    assert [call["value"] for call in client.score_calls] == [4, 2]


@pytest.mark.asyncio
async def test_feedback_reuses_existing_random_score_id(monkeypatch) -> None:
    client = FakeFeedbackClient(
        scores=[SimpleNamespace(id="legacy-random-id", name="user_feedback")]
    )
    monkeypatch.setattr(
        "app.services.feedback_service.get_langfuse_client",
        lambda _settings: client,
    )

    result = await _service().submit(_request(), tenant_id="tenant-a")

    assert result.feedback_id == "legacy-random-id"
    assert client.score_calls[0]["id"] == "legacy-random-id"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tenant_id", "log_id", "client_tenant", "client_log"),
    [
        ("tenant-b", "log-a", "tenant-a", "log-a"),
        ("tenant-a", "log-b", "tenant-a", "log-a"),
        ("tenant-a", "log-a", None, None),
    ],
)
async def test_feedback_rejects_missing_or_mismatched_trace_metadata(
    monkeypatch,
    tenant_id: str,
    log_id: str,
    client_tenant: str | None,
    client_log: str | None,
) -> None:
    client = FakeFeedbackClient(tenant_id=client_tenant, log_id=client_log)
    monkeypatch.setattr(
        "app.services.feedback_service.get_langfuse_client",
        lambda _settings: client,
    )

    with pytest.raises(AppError) as exception_info:
        await _service().submit(_request(log_id=log_id), tenant_id=tenant_id)

    assert exception_info.value.code == ErrorCode.FEEDBACK_LOG_MISMATCH.code
    assert client.score_calls == []


def test_feedback_request_requires_non_empty_log_id() -> None:
    with pytest.raises(ValidationError):
        FeedbackRequest.model_validate({"trace_id": "trace-a", "score": 4})
    with pytest.raises(ValidationError):
        FeedbackRequest.model_validate({"trace_id": "trace-a", "log_id": " ", "score": 4})


@pytest.mark.asyncio
async def test_feedback_rejects_when_langfuse_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.feedback_service.get_langfuse_client",
        lambda _settings: None,
    )

    with pytest.raises(AppError) as exception_info:
        await _service(Settings(langfuse_enabled=False)).submit(
            _request(),
            tenant_id="tenant-a",
        )

    assert exception_info.value.code == ErrorCode.FEEDBACK_UNAVAILABLE.code


@pytest.mark.asyncio
async def test_feedback_returns_unavailable_when_langfuse_write_fails(monkeypatch) -> None:
    client = FakeFeedbackClient(fail_on_flush=True)
    monkeypatch.setattr(
        "app.services.feedback_service.get_langfuse_client",
        lambda _settings: client,
    )

    with pytest.raises(AppError) as exception_info:
        await _service().submit(_request(), tenant_id="tenant-a")

    assert exception_info.value.code == ErrorCode.FEEDBACK_UNAVAILABLE.code


@pytest.mark.asyncio
async def test_feedback_service_defensively_rejects_out_of_range_score(monkeypatch) -> None:
    client = FakeFeedbackClient()
    monkeypatch.setattr(
        "app.services.feedback_service.get_langfuse_client",
        lambda _settings: client,
    )

    with pytest.raises(AppError) as exception_info:
        await _service().submit(
            SimpleNamespace(trace_id="trace-a", log_id="log-a", score=6, comment=None),
            tenant_id="tenant-a",
        )

    assert exception_info.value.code == ErrorCode.FEEDBACK_SCORE_INVALID.code
