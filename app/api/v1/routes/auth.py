from dataclasses import asdict

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_rate_limit_service
from app.api.v1.deps import get_current_tenant
from app.core.auth import TenantContext
from app.core.logging import log_api_call
from app.db.session import get_db
from app.repositories.knowledge_base_repository import KnowledgeBaseRepository
from app.schemas.auth import AuthMeFeatures, AuthMeLimits, AuthMeResponse, AuthMeUsage
from app.schemas.common import APIResponse
from app.services.rate_limit_service import RateLimitService
from app.tenant.plan_resolver import resolve_plan

router = APIRouter(prefix="/auth", tags=["auth"])


@router.get("/me", response_model=APIResponse[AuthMeResponse])
@log_api_call
async def get_current_tenant_info(
    tenant: TenantContext = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db),
    rate_limit_service: RateLimitService = Depends(get_rate_limit_service),
) -> APIResponse[AuthMeResponse]:
    plan = resolve_plan(tenant.plan)
    kb_count = 0
    retrieve_daily_count = 0
    if hasattr(session, "execute"):
        kb_count = await KnowledgeBaseRepository(session).count_by_tenant(
            tenant_id=tenant.tenant_id
        )
        try:
            retrieve_daily_count = await rate_limit_service.get_retrieve_daily_count(
                tenant.tenant_id
            )
        except Exception:
            retrieve_daily_count = 0

    return APIResponse(
        data=AuthMeResponse(
            tenant_id=tenant.tenant_id,
            tenant_name=tenant.tenant_name,
            key_prefix=tenant.key_prefix,
            key_name=tenant.key_name,
            plan=plan.plan,
            features=AuthMeFeatures(**asdict(plan.features)),
            limits=AuthMeLimits(**asdict(plan.limits)),
            usage=AuthMeUsage(
                kb_count=kb_count,
                retrieve_daily_count=retrieve_daily_count,
            ),
        )
    )
