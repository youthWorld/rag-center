from fastapi import APIRouter, Depends

from app.api.v1.deps import get_current_tenant
from app.core.auth import TenantContext
from app.core.logging import log_api_call
from app.schemas.auth import AuthMeResponse
from app.schemas.common import APIResponse

router = APIRouter(prefix="/auth", tags=["auth"])


@router.get("/me", response_model=APIResponse[AuthMeResponse])
@log_api_call
async def get_current_tenant_info(
    tenant: TenantContext = Depends(get_current_tenant),
) -> APIResponse[AuthMeResponse]:
    return APIResponse(
        data=AuthMeResponse(
            tenant_id=tenant.tenant_id,
            tenant_name=tenant.tenant_name,
            key_prefix=tenant.key_prefix,
            key_name=tenant.key_name,
        )
    )
