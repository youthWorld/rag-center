from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from redis.asyncio import Redis

from app.core.config import settings
from app.core.error_codes import ErrorCode
from app.core.exceptions import raise_app_error
from app.tenant.plan_resolver import PlanContext, PlanLimits


class RateLimitService:
    """Apply tenant retrieval limits with the Redis used by Celery."""

    def __init__(
        self,
        redis_client: Any | None = None,
        *,
        redis_url: str | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.redis = redis_client or Redis.from_url(
            redis_url or settings.celery_broker_url,
            decode_responses=True,
            socket_connect_timeout=0.5,
            socket_timeout=0.5,
        )
        self.clock = clock or (lambda: datetime.now(UTC))

    async def check_retrieve(
        self,
        tenant_id: str,
        plan: PlanContext | PlanLimits,
    ) -> None:
        limits = _get_limits(plan)
        daily_count = await self.get_retrieve_daily_count(tenant_id)
        if daily_count >= limits.retrieve_daily:
            raise_app_error(
                ErrorCode.QUOTA_EXCEEDED,
                f"daily retrieval quota exceeded: maximum {limits.retrieve_daily} requests",
                data={
                    "tenant_id": tenant_id,
                    "limit": limits.retrieve_daily,
                    "usage": daily_count,
                },
            )

        second = int(self.clock().timestamp())
        key = f"rag:ratelimit:retrieve:{tenant_id}:{second}"
        current = int(await self.redis.incr(key))
        if current == 1:
            await self.redis.expire(key, 2)
        if current > limits.retrieve_qps:
            raise_app_error(
                ErrorCode.API_RATE_LIMIT,
                f"retrieval rate limit exceeded: maximum {limits.retrieve_qps} requests per second",
                data={
                    "tenant_id": tenant_id,
                    "limit": limits.retrieve_qps,
                },
            )

    async def record_retrieve_success(self, tenant_id: str) -> None:
        key = f"rag:quota:retrieve:daily:{tenant_id}:{self.clock().strftime('%Y%m%d')}"
        current = int(await self.redis.incr(key))
        if current == 1:
            await self.redis.expire(key, 25 * 60 * 60)

    async def get_retrieve_daily_count(self, tenant_id: str) -> int:
        key = f"rag:quota:retrieve:daily:{tenant_id}:{self.clock().strftime('%Y%m%d')}"
        value = await self.redis.get(key)
        return int(value or 0)

    async def close(self) -> None:
        close = getattr(self.redis, "aclose", None) or getattr(self.redis, "close", None)
        if close is not None:
            result = close()
            if hasattr(result, "__await__"):
                await result


def _get_limits(plan: PlanContext | PlanLimits) -> PlanLimits:
    return plan.limits if isinstance(plan, PlanContext) else plan
