from fastapi import APIRouter, Depends, Query, Request
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from starlette.datastructures import UploadFile

from app.api.dependencies import get_document_service, get_settings
from app.api.v1.deps import get_current_tenant
from app.core.auth import TenantContext
from app.core.config import Settings
from app.core.error_codes import ErrorCode
from app.core.exceptions import raise_app_error
from app.core.logging import log_api_call
from app.schemas.common import APIResponse
from app.schemas.document import (
    DocumentDeleteResponse,
    DocumentDetailResponse,
    DocumentUploadRequest,
    DocumentUploadResponse,
)
from app.services.document_service import DocumentService

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=APIResponse[DocumentUploadResponse])
@log_api_call
async def upload_document(
    request: Request,
    tenant: TenantContext = Depends(get_current_tenant),
    service: DocumentService = Depends(get_document_service),
    app_settings: Settings = Depends(get_settings),
) -> APIResponse[DocumentUploadResponse]:
    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if content_type == "application/json":
        try:
            payload = await request.json()
            upload_request = DocumentUploadRequest.model_validate(payload)
        except ValidationError as exc:
            raise RequestValidationError(exc.errors()) from exc
        except ValueError:
            raise_app_error(ErrorCode.PARAM_ERROR, "request body must be valid JSON")
        return APIResponse(data=await service.upload(upload_request, tenant_id=tenant.tenant_id))

    if content_type == "multipart/form-data":
        form = await request.form()
        upload_file = form.get("file")
        if not isinstance(upload_file, UploadFile):
            raise_app_error(ErrorCode.PARAM_ERROR, "file is required")
        kb_id = str(form.get("kb_id") or "").strip()
        if not kb_id or len(kb_id) > 36:
            raise_app_error(ErrorCode.PARAM_ERROR, "kb_id must be a non-empty value")
        title_value = str(form.get("title") or "").strip()
        title = title_value or None
        max_bytes = app_settings.document_max_size_mb * 1024 * 1024
        try:
            file_bytes = await upload_file.read(max_bytes + 1)
        finally:
            await upload_file.close()
        return APIResponse(
            data=await service.upload_file(
                tenant_id=tenant.tenant_id,
                kb_id=kb_id,
                title=title,
                filename=upload_file.filename or "",
                mime_type=upload_file.content_type,
                file_bytes=file_bytes,
            )
        )

    raise_app_error(
        ErrorCode.PARAM_ERROR,
        "Content-Type must be application/json or multipart/form-data",
    )


@router.get("/{document_id}", response_model=APIResponse[DocumentDetailResponse])
@log_api_call
async def get_document(
    document_id: str,
    tenant: TenantContext = Depends(get_current_tenant),
    service: DocumentService = Depends(get_document_service),
) -> APIResponse[DocumentDetailResponse]:
    return APIResponse(data=await service.get(document_id, tenant_id=tenant.tenant_id))


@router.delete("/{document_id}", response_model=APIResponse[DocumentDeleteResponse])
@log_api_call
async def delete_document(
    document_id: str,
    tenant: TenantContext = Depends(get_current_tenant),
    service: DocumentService = Depends(get_document_service),
) -> APIResponse[DocumentDeleteResponse]:
    return APIResponse(data=await service.delete(document_id, tenant_id=tenant.tenant_id))


@router.post(
    "/{document_id}/reindex",
    response_model=APIResponse[DocumentUploadResponse],
)
@log_api_call
async def reindex_document(
    document_id: str,
    reparse: bool = Query(default=False),
    tenant: TenantContext = Depends(get_current_tenant),
    service: DocumentService = Depends(get_document_service),
) -> APIResponse[DocumentUploadResponse]:
    return APIResponse(
        data=await service.reindex(
            document_id,
            tenant_id=tenant.tenant_id,
            reparse=reparse,
        )
    )
