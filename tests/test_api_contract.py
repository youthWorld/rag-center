from datetime import UTC, datetime

import httpx
import pytest

from app.api.dependencies import (
    get_document_service,
    get_knowledge_base_service,
    get_rag_service,
)
from app.api.v1.deps import get_current_tenant
from app.core.auth import TenantContext
from app.main import app
from app.schemas.document import DocumentUploadResponse
from app.schemas.knowledge_base import KnowledgeBaseResponse
from app.schemas.rag import RagRetrieveResponse, RetrievedChunk


class FakeKnowledgeBaseService:
    async def create(self, _request, *, tenant_id: str) -> KnowledgeBaseResponse:
        return KnowledgeBaseResponse(
            kb_id="kb-test",
            name="Test KB",
            tenant_id=tenant_id,
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        )

    async def list_tree(self, *, tenant_id: str, keyword=None):
        del tenant_id
        return []


class FakeDocumentService:
    async def upload(self, _request, *, tenant_id: str) -> DocumentUploadResponse:
        del tenant_id
        return DocumentUploadResponse(
            document_id="document-test",
            kb_id="kb-test",
            status=1,
            chunk_count=2,
        )


class FakeRagService:
    async def retrieve(self, request, *, tenant_id: str) -> RagRetrieveResponse:
        del tenant_id
        evidence_enabled = bool(
            request.evidence_options is not None and request.evidence_options.enabled
        )
        return RagRetrieveResponse(
            query=request.query,
            kb_id=request.kb_id,
            retrieved_chunks=[
                RetrievedChunk(
                    document_id="document-test",
                    chunk_id="chunk-test",
                    title="Test",
                    content="Relevant context",
                    score=0.9,
                )
            ],
            evidence_pack=(
                {
                    "status": "complete",
                    "missing_aspects": [],
                    "groups": [
                        {
                            "aspect": "answer",
                            "covered": True,
                            "evidence": [{"evidence_id": "E1", "role": "core"}],
                        }
                    ],
                    "items": [
                        {
                            "evidence_id": "E1",
                            "chunk_id": "chunk-test",
                            "kb_id": "kb-test",
                            "document_id": "document-test",
                            "title": "Test",
                            "index_version": "v2",
                            "content": "Relevant context",
                            "source": "hybrid",
                            "retrieved_rank": 1,
                        }
                    ],
                }
                if evidence_enabled
                else None
            ),
            metadata={
                "top_k": 5,
                "vector_store": "pgvector",
                "evidence": {
                    "enabled": evidence_enabled,
                    "executed": evidence_enabled,
                    "degraded": False,
                },
            },
        )


@pytest.fixture(autouse=True)
def override_services():
    app.dependency_overrides[get_current_tenant] = lambda: TenantContext(
        tenant_id="tenant-test",
        tenant_name="Test tenant",
        key_id="key-test",
        key_prefix="rk_test",
        key_name="test",
    )
    app.dependency_overrides[get_knowledge_base_service] = lambda: FakeKnowledgeBaseService()
    app.dependency_overrides[get_document_service] = lambda: FakeDocumentService()
    app.dependency_overrides[get_rag_service] = lambda: FakeRagService()
    yield
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_business_routes_are_exposed() -> None:
    assert sorted(app.openapi()["paths"]) == [
        "/api/v1/auth/me",
        "/api/v1/documents/upload",
        "/api/v1/documents/{document_id}",
        "/api/v1/documents/{document_id}/reindex",
        "/api/v1/knowledge-bases/create",
        "/api/v1/knowledge-bases/tree",
        "/api/v1/knowledge-bases/{kb_id}",
        "/api/v1/knowledge-bases/{kb_id}/index-versions",
        "/api/v1/knowledge-bases/{kb_id}/index-versions/rebuild",
        "/api/v1/knowledge-bases/{kb_id}/index-versions/{version}",
        "/api/v1/knowledge-bases/{kb_id}/index-versions/{version}/activate",
        "/api/v1/rag/feedback",
        "/api/v1/rag/retrieve",
    ]


@pytest.mark.asyncio
async def test_business_routes_use_uniform_success_response() -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        create_response = await client.post(
            "/api/v1/knowledge-bases/create",
            json={"name": "Test KB", "description": "desc", "tenant_id": "ignored"},
        )
        upload_response = await client.post(
            "/api/v1/documents/upload",
            json={
                "kb_id": "kb-test",
                "title": "Test",
                "content": "content",
            },
        )
        retrieve_response = await client.post(
            "/api/v1/rag/retrieve",
            json={
                "kb_id": "kb-test",
                "user_id": "user-test",
                "query": "question",
            },
        )
        tree_response = await client.get(
            "/api/v1/knowledge-bases/tree",
            params={"keyword": "tech"},
        )

    assert create_response.status_code == 200
    assert upload_response.status_code == 200
    assert retrieve_response.status_code == 200
    assert tree_response.status_code == 200
    assert create_response.json()["code"] == 0
    assert upload_response.json()["code"] == 0
    assert retrieve_response.json()["code"] == 0
    assert tree_response.json() == {"code": 0, "msg": "success", "data": []}
    assert retrieve_response.json()["data"]["retrieved_chunks"][0]["score"] == 0.9
    assert retrieve_response.json()["data"]["evidence_pack"] is None


@pytest.mark.asyncio
async def test_retrieve_contract_accepts_and_returns_evidence_pack() -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/rag/retrieve",
            json={
                "kb_id": "kb-test",
                "user_id": "user-test",
                "query": "question",
                "evidence_options": {"enabled": True, "max_items": 20},
            },
        )

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["evidence_pack"]["items"][0]["evidence_id"] == "E1"
    assert data["metadata"]["evidence"]["executed"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("max_items", [0, 21])
async def test_retrieve_contract_rejects_invalid_evidence_max_items(max_items: int) -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/rag/retrieve",
            json={
                "kb_id": "kb-test",
                "user_id": "user-test",
                "query": "question",
                "evidence_options": {"enabled": True, "max_items": max_items},
            },
        )

    assert response.status_code == 400
    assert response.json()["code"] == 20004
