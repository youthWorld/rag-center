from fastapi import APIRouter, Depends

from app.api.dependencies import get_rag_service
from app.core.logging import log_api_call
from app.schemas.common import APIResponse
from app.schemas.rag import RagRetrieveRequest, RagRetrieveResponse
from app.services.rag_service import RagService

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/retrieve", response_model=APIResponse[RagRetrieveResponse])
@log_api_call
async def retrieve(
    request: RagRetrieveRequest,
    service: RagService = Depends(get_rag_service),
) -> APIResponse[RagRetrieveResponse]:
    return APIResponse(data=await service.retrieve(request))
