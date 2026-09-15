from fastapi import Depends, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_settings
from app.core.auth import (
    DEFAULT_TENANT_ID,
    DEFAULT_TENANT_NAME,
    TenantContext,
    authenticate_api_key,
)
from app.core.config import Settings
from app.core.error_codes import ErrorCode
from app.core.exceptions import AppError
from app.db.session import get_db

bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_tenant(
    credentials: HTTPAuthorizationCredentials | None = Security(bearer_scheme),
    app_settings: Settings = Depends(get_settings),
    session: AsyncSession = Depends(get_db),
) -> TenantContext:
    if not app_settings.auth_enabled:
        return TenantContext(
            tenant_id=DEFAULT_TENANT_ID,
            tenant_name=DEFAULT_TENANT_NAME,
            key_id=None,
            key_prefix=None,
            key_name=None,
        )

    if credentials is None or credentials.scheme.lower() != "bearer":
        raise AppError(
            code=ErrorCode.UNAUTHORIZED,
            internal_message="Bearer API key is required",
            context={"auth_failure": "missing bearer credentials"},
        )

    return await authenticate_api_key(session, credentials.credentials)
