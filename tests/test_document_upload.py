import httpx
import pytest

from app.api.dependencies import get_document_service
from app.api.v1.deps import get_current_tenant
from app.core.auth import TenantContext
from app.main import app
from app.schemas.document import DocumentUploadResponse


class FakeMultipartDocumentService:
    async def upload_file(self, **kwargs) -> DocumentUploadResponse:
        assert kwargs["kb_id"] == "kb-test"
        assert kwargs["filename"] == "sample.md"
        assert kwargs["file_bytes"] == b"# sample"
        return DocumentUploadResponse(
            document_id="document-file",
            kb_id="kb-test",
            status=3,
            chunk_count=0,
        )


@pytest.mark.asyncio
async def test_multipart_upload_uses_file_service() -> None:
    app.dependency_overrides[get_current_tenant] = lambda: TenantContext(
        tenant_id="tenant-test",
        tenant_name="Test tenant",
        key_id="key-test",
        key_prefix="rk_test",
        key_name="test",
    )
    app.dependency_overrides[get_document_service] = lambda: FakeMultipartDocumentService()
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/documents/upload",
                data={"kb_id": "kb-test"},
                files={"file": ("sample.md", b"# sample", "text/markdown")},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["data"]["document_id"] == "document-file"
