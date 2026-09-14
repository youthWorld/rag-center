from fastapi import APIRouter, Depends

from app.api.dependencies import get_knowledge_base_service
from app.schemas.common import APIResponse
from app.schemas.knowledge_base import KnowledgeBaseCreateRequest, KnowledgeBaseResponse
from app.services.knowledge_base_service import KnowledgeBaseService

router = APIRouter(prefix="/knowledge-bases", tags=["knowledge-bases"])


@router.post("/create", response_model=APIResponse[KnowledgeBaseResponse])
async def create_knowledge_base(
    request: KnowledgeBaseCreateRequest,
    service: KnowledgeBaseService = Depends(get_knowledge_base_service),
) -> APIResponse[KnowledgeBaseResponse]:
    return APIResponse(data=await service.create(request))
