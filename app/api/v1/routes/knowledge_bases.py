from fastapi import APIRouter, Depends, Query

from app.api.dependencies import get_knowledge_base_service
from app.core.logging import log_api_call
from app.schemas.common import APIResponse
from app.schemas.knowledge_base import (
    KnowledgeBaseCreateRequest,
    KnowledgeBaseResponse,
    KnowledgeBaseTenantTreeResponse,
)
from app.services.knowledge_base_service import KnowledgeBaseService

router = APIRouter(prefix="/knowledge-bases", tags=["knowledge-bases"])


@router.post("/create", response_model=APIResponse[KnowledgeBaseResponse])
@log_api_call
async def create_knowledge_base(
    request: KnowledgeBaseCreateRequest,
    service: KnowledgeBaseService = Depends(get_knowledge_base_service),
) -> APIResponse[KnowledgeBaseResponse]:
    return APIResponse(data=await service.create(request))


@router.get("/tree", response_model=APIResponse[list[KnowledgeBaseTenantTreeResponse]])
@log_api_call
async def list_knowledge_base_tree(
    keyword: str | None = Query(default=None, max_length=128),
    service: KnowledgeBaseService = Depends(get_knowledge_base_service),
) -> APIResponse[list[KnowledgeBaseTenantTreeResponse]]:
    return APIResponse(data=await service.list_tree(keyword=keyword))
