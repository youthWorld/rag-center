from fastapi import APIRouter, Depends, Query

from app.api.dependencies import get_index_version_service, get_knowledge_base_service
from app.api.v1.deps import get_current_tenant
from app.core.auth import TenantContext
from app.core.logging import log_api_call
from app.schemas.common import APIResponse
from app.schemas.index_version import (
    IndexVersionListResponse,
    IndexVersionRebuildResponse,
    IndexVersionResponse,
)
from app.schemas.knowledge_base import (
    KnowledgeBaseCreateRequest,
    KnowledgeBaseDeleteResponse,
    KnowledgeBaseDetailResponse,
    KnowledgeBaseResponse,
    KnowledgeBaseTenantTreeResponse,
    KnowledgeBaseUpdateRequest,
)
from app.services.index_version_service import IndexVersionService
from app.services.knowledge_base_service import KnowledgeBaseService

router = APIRouter(prefix="/knowledge-bases", tags=["knowledge-bases"])


@router.post(
    "/{kb_id}/index-versions/rebuild",
    response_model=APIResponse[IndexVersionRebuildResponse],
)
@log_api_call
async def rebuild_index_version(
    kb_id: str,
    version: str = Query(default="v2", min_length=1, max_length=32),
    tenant: TenantContext = Depends(get_current_tenant),
    service: IndexVersionService = Depends(get_index_version_service),
) -> APIResponse[IndexVersionRebuildResponse]:
    return APIResponse(
        data=await service.rebuild(
            kb_id=kb_id,
            tenant_id=tenant.tenant_id,
            version=version,
        )
    )


@router.get(
    "/{kb_id}/index-versions",
    response_model=APIResponse[IndexVersionListResponse],
)
@log_api_call
async def list_index_versions(
    kb_id: str,
    tenant: TenantContext = Depends(get_current_tenant),
    service: IndexVersionService = Depends(get_index_version_service),
) -> APIResponse[IndexVersionListResponse]:
    return APIResponse(data=await service.list_versions(kb_id=kb_id, tenant_id=tenant.tenant_id))


@router.post(
    "/{kb_id}/index-versions/{version}/activate",
    response_model=APIResponse[IndexVersionResponse],
)
@log_api_call
async def activate_index_version(
    kb_id: str,
    version: str,
    tenant: TenantContext = Depends(get_current_tenant),
    service: IndexVersionService = Depends(get_index_version_service),
) -> APIResponse[IndexVersionResponse]:
    return APIResponse(
        data=await service.activate(
            kb_id=kb_id,
            tenant_id=tenant.tenant_id,
            version=version,
        )
    )


@router.delete(
    "/{kb_id}/index-versions/{version}",
    response_model=APIResponse[dict[str, str]],
)
@log_api_call
async def delete_index_version(
    kb_id: str,
    version: str,
    tenant: TenantContext = Depends(get_current_tenant),
    service: IndexVersionService = Depends(get_index_version_service),
) -> APIResponse[dict[str, str]]:
    return APIResponse(
        data=await service.delete(
            kb_id=kb_id,
            tenant_id=tenant.tenant_id,
            version=version,
        )
    )


@router.post("/create", response_model=APIResponse[KnowledgeBaseResponse])
@log_api_call
async def create_knowledge_base(
    request: KnowledgeBaseCreateRequest,
    tenant: TenantContext = Depends(get_current_tenant),
    service: KnowledgeBaseService = Depends(get_knowledge_base_service),
) -> APIResponse[KnowledgeBaseResponse]:
    return APIResponse(data=await service.create(request, tenant_id=tenant.tenant_id))


@router.get("/tree", response_model=APIResponse[list[KnowledgeBaseTenantTreeResponse]])
@log_api_call
async def list_knowledge_base_tree(
    keyword: str | None = Query(default=None, max_length=128),
    tenant: TenantContext = Depends(get_current_tenant),
    service: KnowledgeBaseService = Depends(get_knowledge_base_service),
) -> APIResponse[list[KnowledgeBaseTenantTreeResponse]]:
    return APIResponse(data=await service.list_tree(tenant_id=tenant.tenant_id, keyword=keyword))


@router.get("/{kb_id}", response_model=APIResponse[KnowledgeBaseDetailResponse])
@log_api_call
async def get_knowledge_base(
    kb_id: str,
    tenant: TenantContext = Depends(get_current_tenant),
    service: KnowledgeBaseService = Depends(get_knowledge_base_service),
) -> APIResponse[KnowledgeBaseDetailResponse]:
    return APIResponse(data=await service.get(kb_id, tenant_id=tenant.tenant_id))


@router.patch("/{kb_id}", response_model=APIResponse[KnowledgeBaseDetailResponse])
@log_api_call
async def update_knowledge_base(
    kb_id: str,
    request: KnowledgeBaseUpdateRequest,
    tenant: TenantContext = Depends(get_current_tenant),
    service: KnowledgeBaseService = Depends(get_knowledge_base_service),
) -> APIResponse[KnowledgeBaseDetailResponse]:
    return APIResponse(data=await service.update(kb_id, request, tenant_id=tenant.tenant_id))


@router.delete("/{kb_id}", response_model=APIResponse[KnowledgeBaseDeleteResponse])
@log_api_call
async def delete_knowledge_base(
    kb_id: str,
    tenant: TenantContext = Depends(get_current_tenant),
    service: KnowledgeBaseService = Depends(get_knowledge_base_service),
) -> APIResponse[KnowledgeBaseDeleteResponse]:
    return APIResponse(data=await service.delete(kb_id, tenant_id=tenant.tenant_id))
