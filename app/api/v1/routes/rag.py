from fastapi import APIRouter, Depends

from app.api.dependencies import get_feedback_service, get_rag_service
from app.api.v1.deps import get_current_tenant
from app.core.auth import TenantContext
from app.core.logging import log_api_call
from app.schemas.common import APIResponse
from app.schemas.feedback import FeedbackData, FeedbackRequest
from app.schemas.rag import RagRetrieveRequest, RagRetrieveResponse
from app.services.feedback_service import FeedbackService
from app.services.rag_service import RagService

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/retrieve", response_model=APIResponse[RagRetrieveResponse])
@log_api_call
async def retrieve(
    request: RagRetrieveRequest,
    tenant: TenantContext = Depends(get_current_tenant),
    service: RagService = Depends(get_rag_service),
) -> APIResponse[RagRetrieveResponse]:
    return APIResponse(data=await service.retrieve(request, tenant_id=tenant.tenant_id))


@router.post("/feedback", response_model=APIResponse[FeedbackData])
@log_api_call
async def submit_feedback(
    request: FeedbackRequest,
    tenant: TenantContext = Depends(get_current_tenant),
    service: FeedbackService = Depends(get_feedback_service),
) -> APIResponse[FeedbackData]:
    return APIResponse(data=await service.submit(request, tenant_id=tenant.tenant_id))
