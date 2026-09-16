from __future__ import annotations

import hashlib
import re
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import NoReturn

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.error_codes import ErrorCode
from app.core.exceptions import AppError
from app.models.api_key import ApiKey
from app.repositories.api_key_repository import ApiKeyRepository

API_KEY_PREFIX = "rk_live_"
DEFAULT_TENANT_ID = "tenant_demo"
DEFAULT_TENANT_NAME = "tenant_demo"
_API_KEY_PATTERN = re.compile(r"^rk_live_[0-9a-f]{32}$")


@dataclass(frozen=True, slots=True)
class TenantContext:
    tenant_id: str
    tenant_name: str
    key_id: str | None
    key_prefix: str | None
    key_name: str | None = None
    plan: str = "free"


def generate_api_key() -> tuple[str, str, str]:
    """Return plaintext key, SHA-256 hash, and its safe log prefix."""

    plaintext = f"{API_KEY_PREFIX}{secrets.token_hex(16)}"
    return plaintext, hash_api_key(plaintext), get_key_prefix(plaintext)


def hash_api_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def get_key_prefix(api_key: str) -> str:
    return f"rk_{api_key[len(API_KEY_PREFIX):len(API_KEY_PREFIX) + 4]}"


async def authenticate_api_key(session: AsyncSession, api_key: str) -> TenantContext:
    if not _API_KEY_PATTERN.fullmatch(api_key):
        _raise_unauthorized("API key format is invalid")

    record = await ApiKeyRepository(session).get_by_hash(hash_api_key(api_key))
    if record is None:
        _raise_unauthorized("API key was not found")
    if record.status != "active":
        _raise_unauthorized("API key is not active")
    if _is_expired(record):
        _raise_unauthorized("API key has expired")

    tenant = record.tenant
    if tenant is None or tenant.status != "active":
        _raise_unauthorized("tenant is not active")

    return TenantContext(
        tenant_id=tenant.id,
        tenant_name=tenant.name,
        key_id=record.id,
        key_prefix=record.key_prefix,
        key_name=record.name,
        plan=getattr(tenant, "plan", "free"),
    )


def _is_expired(record: ApiKey) -> bool:
    if record.expires_at is None:
        return False
    expires_at = record.expires_at
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=UTC)
    return expires_at <= datetime.now(UTC)


def _raise_unauthorized(reason: str) -> NoReturn:
    raise AppError(
        code=ErrorCode.UNAUTHORIZED,
        internal_message=reason,
        context={"auth_failure": reason},
    )
