from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from app.api.dependencies import get_document_service, get_knowledge_base_service
from app.api.v1.deps import get_current_tenant
from app.core.auth import TenantContext
from app.core.error_codes import ErrorCode
from app.main import app
from app.services.document_service import DocumentService
from app.services.knowledge_base_service import KnowledgeBaseService

TENANT_B = "tenant-b"
KB_A = "kb-owned-by-a"
DOCUMENT_A = "document-owned-by-a"


@pytest.fixture
def tenant_b_services():
    knowledge_base_repository = SimpleNamespace(
        get_by_id_and_tenant=AsyncMock(return_value=None),
    )
    document_repository = SimpleNamespace(
        get_by_id_and_tenant=AsyncMock(return_value=None),
    )
    knowledge_base_service = KnowledgeBaseService(
        session=SimpleNamespace(),
        repository=knowledge_base_repository,
        document_repository=SimpleNamespace(),
        indexing_service=SimpleNamespace(),
    )
    document_service = DocumentService(
        session=SimpleNamespace(),
        document_repository=document_repository,
        knowledge_base_repository=SimpleNamespace(),
        indexing_service=SimpleNamespace(),
    )
    return knowledge_base_service, document_service


@pytest.mark.asyncio
async def test_cross_tenant_resources_are_indistinguishable_from_missing_resources(
    tenant_b_services,
) -> None:
    knowledge_base_service, document_service = tenant_b_services
    previous_overrides = app.dependency_overrides.copy()
    app.dependency_overrides[get_current_tenant] = lambda: TenantContext(
        tenant_id=TENANT_B,
        tenant_name=TENANT_B,
        key_id="key-b",
        key_prefix="rk_bbbb",
        key_name="Key B",
    )
    app.dependency_overrides[get_knowledge_base_service] = lambda: knowledge_base_service
    app.dependency_overrides[get_document_service] = lambda: document_service

    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            cross_tenant_responses = [
                await client.get(f"/api/v1/knowledge-bases/{KB_A}"),
                await client.patch(
                    f"/api/v1/knowledge-bases/{KB_A}",
                    json={"name": "unauthorized update"},
                ),
                await client.delete(f"/api/v1/knowledge-bases/{KB_A}"),
                await client.get(f"/api/v1/documents/{DOCUMENT_A}"),
                await client.delete(f"/api/v1/documents/{DOCUMENT_A}"),
                await client.post(f"/api/v1/documents/{DOCUMENT_A}/reindex"),
            ]
            missing_responses = [
                await client.get("/api/v1/knowledge-bases/missing-kb"),
                await client.patch(
                    "/api/v1/knowledge-bases/missing-kb",
                    json={"name": "unauthorized update"},
                ),
                await client.delete("/api/v1/knowledge-bases/missing-kb"),
                await client.get("/api/v1/documents/missing-document"),
                await client.delete("/api/v1/documents/missing-document"),
                await client.post("/api/v1/documents/missing-document/reindex"),
            ]
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(previous_overrides)

    assert [response.status_code for response in cross_tenant_responses] == [404] * 6
    assert [response.json() for response in cross_tenant_responses] == [
        response.json() for response in missing_responses
    ]
    assert all(
        response.json()["code"] == ErrorCode.NOT_FOUND.code
        for response in cross_tenant_responses
    )
    knowledge_base_service.repository.get_by_id_and_tenant.assert_any_await(
        kb_id=KB_A,
        tenant_id=TENANT_B,
    )
    document_service.document_repository.get_by_id_and_tenant.assert_any_await(
        document_id=DOCUMENT_A,
        tenant_id=TENANT_B,
    )
