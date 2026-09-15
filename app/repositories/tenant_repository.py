from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.tenant import Tenant


class TenantRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_by_id(self, tenant_id: str) -> Tenant | None:
        result = await self.session.execute(select(Tenant).where(Tenant.id == tenant_id))
        return result.scalar_one_or_none()

    async def create(self, *, tenant_id: str, name: str) -> Tenant:
        tenant = Tenant(id=tenant_id, name=name)
        self.session.add(tenant)
        await self.session.flush()
        return tenant
