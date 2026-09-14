from fastapi import APIRouter, Depends

from app.api.dependencies import get_document_service
from app.schemas.common import APIResponse
from app.schemas.document import DocumentUploadRequest, DocumentUploadResponse
from app.services.document_service import DocumentService

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=APIResponse[DocumentUploadResponse])
async def upload_document(
    request: DocumentUploadRequest,
    service: DocumentService = Depends(get_document_service),
) -> APIResponse[DocumentUploadResponse]:
    return APIResponse(data=await service.upload(request))
