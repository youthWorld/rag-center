import httpx
import pytest

from app.api.dependencies import get_research_service
from app.api.v1.deps import get_current_tenant
from app.core.auth import TenantContext
from app.main import app
from app.schemas.research import ResearchData


class FakeResearchService:
    def __init__(self) -> None:
        self.calls = []

    async def research(self, request, *, tenant_id: str, tenant):
        self.calls.append((request, tenant_id, tenant))
        kb_ids = request.resolved_kb_ids()
        return ResearchData.model_validate(
            {
                "query": request.query,
                "kb_id": kb_ids[0],
                "kb_ids": kb_ids,
                "answer": "按原规则处理。[E1]",
                "evidence_pack": {
                    "status": "complete",
                    "missing_aspects": [],
                    "groups": [
                        {
                            "aspect": "规则",
                            "covered": True,
                            "evidence": [{"evidence_id": "E1", "role": "core"}],
                        }
                    ],
                    "items": [
                        {
                            "evidence_id": "E1",
                            "chunk_id": "chunk-a",
                            "kb_id": kb_ids[0],
                            "document_id": "doc-a",
                            "title": "规则",
                            "content": "原文",
                            "index_version": "v1",
                            "source": "hybrid",
                            "rounds": [1],
                            "query_ids": ["Q1"],
                        }
                    ],
                },
                "plan": {
                    "aspects": [{"aspect_id": "A1", "description": "规则"}],
                    "initial_queries": [
                        {"query_id": "Q1", "query": request.query, "aspect_ids": ["A1"]}
                    ],
                },
                "rounds": [{
                    "round": 1, "purpose": "initial", "candidate_count": 1,
                    "new_chunk_count": 1, "latency_ms": 12,
                    "tasks": [{
                        "query_id": "Q1", "query": request.query,
                        "aspect_ids": ["A1"], "round": 1, "success": True,
                        "latency_ms": 12, "chunk_count": 1, "new_chunk_count": 1,
                    }],
                }],
                "metadata": {
                    "research_id": "research-a",
                    "log_id": "log-a",
                    "profile": "research_fixed",
                    "tenant_plan": "pro",
                    "index_versions": {kb_id: "v1" for kb_id in kb_ids},
                    "round_count": 1,
                    "retrieval_task_count": 1,
                    "llm_call_count": 3,
                    "input_tokens": 10,
                    "output_tokens": 10,
                    "latency_ms": 12,
                    "stop_reason": "evidence_complete",
                    "first_round_missing_aspect_ids": [],
                },
            }
        )


@pytest.fixture
def service():
    fake = FakeResearchService()
    app.dependency_overrides[get_current_tenant] = lambda: TenantContext(
        tenant_id="tenant-a",
        tenant_name="Tenant A",
        key_id="key-a",
        key_prefix="rk_a",
        key_name="test",
        plan="pro",
    )
    app.dependency_overrides[get_research_service] = lambda: fake
    yield fake
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_research_route_normalizes_kb_ids_and_exposes_answer_without_chunks(
    service: FakeResearchService,
) -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/v1/rag/research",
            json={
                "kb_ids": [" kb-a ", "kb-a", "kb-b"],
                "user_id": "user-a",
                "query": "  什么规则？  ",
            },
        )
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["kb_ids"] == ["kb-a", "kb-b"]
    assert data["query"] == "什么规则？"
    assert data["answer"] == "按原规则处理。[E1]"
    assert data["evidence_pack"]["items"][0]["rounds"] == [1]
    assert data["metadata"]["tenant_plan"] == "pro"
    assert data["metadata"]["first_round_missing_aspect_ids"] == []
    assert "retrieved_chunks" not in data
    assert "chunks" not in data
    assert data["rounds"][0]["tasks"][0]["chunk_count"] == 1
    assert "chunk_ids" not in data["rounds"][0]["tasks"][0]
    assert service.calls[0][1] == "tenant-a"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extra",
    [{"profile": "quality"}, {"index_version": "v2"}, {"top_k": 5}, {"model": "other"}],
)
async def test_research_route_rejects_client_overrides(
    service: FakeResearchService, extra: dict
) -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/v1/rag/research",
            json={"kb_id": "kb-a", "user_id": "user-a", "query": "问题", **extra},
        )
    assert response.status_code == 400
    assert response.json()["code"] == 20004
    assert service.calls == []


@pytest.mark.asyncio
async def test_research_route_passes_authenticated_tenant_not_client_tenant(
    service: FakeResearchService,
) -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/v1/rag/research",
            json={
                "kb_id": "kb-a", "user_id": "user-a", "query": "问题",
                "tenant_id": "tenant-b",
            },
        )
    assert response.status_code == 400
    assert service.calls == []
