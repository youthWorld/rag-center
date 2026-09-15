from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.models.api_key import ApiKey


class ApiKeyRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_by_hash(self, key_hash: str) -> ApiKey | None:
        statement = (
            select(ApiKey)
            .options(joinedload(ApiKey.tenant))
            .where(ApiKey.key_hash == key_hash)
        )
        result = await self.session.execute(statement)
        return result.scalar_one_or_none()

    async def create(
        self,
        *,
        tenant_id: str,
        key_hash: str,
        key_prefix: str,
        name: str,
        expires_at: datetime | None = None,
    ) -> ApiKey:
        api_key = ApiKey(
            tenant_id=tenant_id,
            key_hash=key_hash,
            key_prefix=key_prefix,
            name=name,
            expires_at=expires_at,
        )
        self.session.add(api_key)
        await self.session.flush()
        return api_key
