from fastapi import APIRouter, Depends

from app.api.dependencies import get_document_service
from app.api.v1.deps import get_current_tenant
from app.core.auth import TenantContext
from app.core.logging import log_api_call
from app.schemas.common import APIResponse
from app.schemas.document import DocumentUploadRequest, DocumentUploadResponse
from app.services.document_service import DocumentService

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=APIResponse[DocumentUploadResponse])
@log_api_call
async def upload_document(
    request: DocumentUploadRequest,
    tenant: TenantContext = Depends(get_current_tenant),
    service: DocumentService = Depends(get_document_service),
) -> APIResponse[DocumentUploadResponse]:
    return APIResponse(data=await service.upload(request, tenant_id=tenant.tenant_id))
