from __future__ import annotations

from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Any

from app.core.error_codes import ErrorCode
from app.core.exceptions import AppError
from app.tenant.plan_presets import get_plan_preset

DEFAULT_FALLBACK_TENANT_ID = "tenant_demo"


@dataclass(frozen=True, slots=True)
class PlanFeatures:
    allowed_profiles: list[str]
    hybrid_allowed: bool
    rerank_allowed: bool
    query_rewrite_allowed: bool


@dataclass(frozen=True, slots=True)
class PlanLimits:
    retrieve_qps: int
    retrieve_daily: int
    max_kb: int
    max_documents_per_kb: int
    max_processing_documents: int


@dataclass(frozen=True, slots=True)
class PlanContext:
    plan: str
    features: PlanFeatures
    limits: PlanLimits

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_plan(tenant: Any) -> PlanContext:
    """Resolve the immutable policy for a tenant model or tenant-like object."""

    plan = tenant if isinstance(tenant, str) else getattr(tenant, "plan", None)
    if not isinstance(plan, str) or not plan.strip():
        raise AppError(
            "tenant plan is not configured",
            code=ErrorCode.CONFIGURATION_ERROR,
            context={"tenant_id": getattr(tenant, "id", None)},
        )

    normalized_plan = plan.strip().lower()
    try:
        preset = get_plan_preset(normalized_plan)
    except ValueError as exc:
        raise AppError(
            "tenant plan is invalid",
            code=ErrorCode.CONFIGURATION_ERROR,
            internal_message=str(exc),
            context={"tenant_id": getattr(tenant, "id", None), "plan": plan},
        ) from exc

    return PlanContext(
        plan=normalized_plan,
        features=PlanFeatures(**preset["features"]),
        limits=PlanLimits(**preset["limits"]),
    )


class PlanResolver:
    """Load a tenant record and convert it to a plan context."""

    def __init__(self, tenant_repository: Any | None = None, *, fallback_plan: str = "pro") -> None:
        self.tenant_repository = tenant_repository
        self.fallback_plan = fallback_plan

    def resolve(self, tenant: Any) -> PlanContext:
        return resolve_plan(tenant)

    def resolve_plan(self, tenant: Any) -> PlanContext:
        return resolve_plan(tenant)

    async def resolve_for_tenant_id(self, tenant_id: str) -> PlanContext:
        if self.tenant_repository is None:
            return resolve_plan(SimpleNamespace(id=tenant_id, plan=self.fallback_plan))

        tenant = await self.tenant_repository.get_by_id(tenant_id)
        if tenant is None:
            if tenant_id == DEFAULT_FALLBACK_TENANT_ID:
                return resolve_plan(
                    SimpleNamespace(id=tenant_id, plan=self.fallback_plan)
                )
            raise AppError(
                "tenant not found",
                code=ErrorCode.NOT_FOUND,
                context={"tenant_id": tenant_id},
            )
        return resolve_plan(tenant)
