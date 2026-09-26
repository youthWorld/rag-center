import httpx
import pytest

from app.api.dependencies import get_feedback_service
from app.api.v1.deps import get_current_tenant
from app.core.auth import TenantContext
from app.main import app
from app.schemas.feedback import FeedbackData


class FakeFeedbackService:
    async def submit(self, request, *, tenant_id: str) -> FeedbackData:
        return FeedbackData(
            feedback_id="feedback-test",
            trace_id=request.trace_id,
            log_id=request.log_id,
            score=request.score,
        )


@pytest.fixture(autouse=True)
def override_feedback_dependencies():
    app.dependency_overrides[get_current_tenant] = lambda: TenantContext(
        tenant_id="tenant-test",
        tenant_name="Test tenant",
        key_id="key-test",
        key_prefix="rk_test",
        key_name="test",
    )
    app.dependency_overrides[get_feedback_service] = lambda: FakeFeedbackService()
    yield
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_feedback_route_returns_uniform_success_envelope() -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/rag/feedback",
            json={
                "trace_id": "trace-test",
                "log_id": "log-test",
                "score": 4,
                "comment": "useful",
                "ignored": "field",
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "code": 0,
        "msg": "success",
        "data": {
            "feedback_id": "feedback-test",
            "trace_id": "trace-test",
            "log_id": "log-test",
            "score": 4,
        },
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("score", [0, 6])
async def test_feedback_score_range_returns_20022(score: int) -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/rag/feedback",
            json={"trace_id": "trace-test", "log_id": "log-test", "score": score},
        )

    assert response.status_code == 400
    assert response.json()["code"] == 20022


@pytest.mark.asyncio
async def test_feedback_rejects_blank_trace_id() -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/rag/feedback",
            json={"trace_id": "   ", "log_id": "log-test", "score": 4},
        )

    assert response.status_code == 400
    assert response.json()["code"] == 20004
