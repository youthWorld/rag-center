from datetime import UTC, datetime
from types import SimpleNamespace

import httpx
import pytest

import app.api.v1.deps as auth_dependencies
from app.api.dependencies import get_settings
from app.core.auth import (
    DEFAULT_TENANT_ID,
    TenantContext,
    authenticate_api_key,
    generate_api_key,
    hash_api_key,
)
from app.core.config import Settings
from app.core.error_codes import ErrorCode
from app.core.exceptions import AppError
from app.db.session import get_db
from app.main import app


def _api_key_record(
    *,
    tenant_status: str = "active",
    key_status: str = "active",
    expires_at: datetime | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id="key-test",
        status=key_status,
        expires_at=expires_at,
        key_prefix="rk_a1b2",
        name="test key",
        tenant=SimpleNamespace(
            id="tenant-test",
            name="Test tenant",
            status=tenant_status,
        ),
    )


@pytest.mark.asyncio
async def test_authenticate_api_key_hashes_plaintext_and_returns_context(monkeypatch) -> None:
    plaintext = "rk_live_" + "a" * 32
    record = _api_key_record()
    repository = SimpleNamespace(
        get_by_hash=lambda key_hash: _async_return(
            key_hash == hash_api_key(plaintext),
            record,
        )
    )

    class FakeApiKeyRepository:
        def __init__(self, _session) -> None:
            pass

        async def get_by_hash(self, key_hash: str):
            return await repository.get_by_hash(key_hash)

    monkeypatch.setattr("app.core.auth.ApiKeyRepository", FakeApiKeyRepository)

    context = await authenticate_api_key(object(), plaintext)

    assert context == TenantContext(
        tenant_id="tenant-test",
        tenant_name="Test tenant",
        key_id="key-test",
        key_prefix="rk_a1b2",
        key_name="test key",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tenant_status", "key_status", "expires_at"),
    [
        ("active", "revoked", None),
        ("disabled", "active", None),
        ("active", "active", datetime(2026, 9, 14, tzinfo=UTC)),
    ],
)
async def test_authenticate_api_key_rejects_inactive_or_expired_records(
    monkeypatch,
    tenant_status: str,
    key_status: str,
    expires_at: datetime | None,
) -> None:
    record = _api_key_record(
        tenant_status=tenant_status,
        key_status=key_status,
        expires_at=expires_at,
    )

    class FakeApiKeyRepository:
        def __init__(self, _session) -> None:
            pass

        async def get_by_hash(self, _key_hash: str):
            return record

    monkeypatch.setattr("app.core.auth.ApiKeyRepository", FakeApiKeyRepository)

    with pytest.raises(AppError) as raised:
        await authenticate_api_key(object(), "rk_live_" + "b" * 32)

    assert raised.value.code == ErrorCode.UNAUTHORIZED.code


def _async_return(matches: bool, record: SimpleNamespace):
    async def result():
        return record if matches else None

    return result()


def _override_auth_settings(enabled: bool) -> None:
    app.dependency_overrides[get_settings] = lambda: Settings(auth_enabled=enabled)
    app.dependency_overrides[get_db] = lambda: object()


@pytest.mark.asyncio
async def test_auth_me_rejects_missing_key() -> None:
    _override_auth_settings(True)
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/v1/auth/me")
    finally:
        app.dependency_overrides.pop(get_settings, None)
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 401
    assert response.json()["code"] == ErrorCode.UNAUTHORIZED.code


@pytest.mark.asyncio
async def test_auth_me_rejects_malformed_key() -> None:
    _override_auth_settings(True)
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/api/v1/auth/me",
                headers={"Authorization": "Bearer invalid-key"},
            )
    finally:
        app.dependency_overrides.pop(get_settings, None)
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 401
    assert response.json()["code"] == ErrorCode.UNAUTHORIZED.code


@pytest.mark.asyncio
async def test_auth_me_returns_tenant_context_for_valid_key(monkeypatch) -> None:
    async def fake_authenticate(_session, _api_key: str) -> TenantContext:
        return TenantContext(
            tenant_id="tenant-test",
            tenant_name="Test tenant",
            key_id="key-test",
            key_prefix="rk_a1b2",
            key_name="test key",
        )

    monkeypatch.setattr(auth_dependencies, "authenticate_api_key", fake_authenticate)
    _override_auth_settings(True)
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/api/v1/auth/me",
                headers={"Authorization": "Bearer rk_live_" + "c" * 32},
            )
    finally:
        app.dependency_overrides.pop(get_settings, None)
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 200
    assert response.json() == {
        "code": 0,
        "msg": "success",
        "data": {
            "tenant_id": "tenant-test",
            "tenant_name": "Test tenant",
            "key_prefix": "rk_a1b2",
            "key_name": "test key",
            "plan": "free",
            "features": {
                "allowed_profiles": ["speed"],
                "hybrid_allowed": False,
                "rerank_allowed": False,
                "query_rewrite_allowed": False,
                "evidence_allowed": False,
                "research_allowed": False,
            },
            "limits": {
                "retrieve_qps": 3,
                "retrieve_daily": 500,
                "max_kb": 1,
                "max_kb_per_retrieve": 1,
                "max_documents_per_kb": 30,
                "max_processing_documents": 1,
            },
            "usage": {"kb_count": 0, "retrieve_daily_count": 0},
        },
    }


@pytest.mark.asyncio
async def test_auth_me_uses_default_tenant_when_auth_is_disabled() -> None:
    _override_auth_settings(False)
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/v1/auth/me")
    finally:
        app.dependency_overrides.pop(get_settings, None)
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 200
    assert response.json()["data"] == {
        "tenant_id": DEFAULT_TENANT_ID,
        "tenant_name": DEFAULT_TENANT_ID,
        "key_prefix": None,
        "key_name": None,
        "plan": "pro",
        "features": {
            "allowed_profiles": ["speed", "balanced", "quality", "custom"],
            "hybrid_allowed": True,
            "rerank_allowed": True,
            "query_rewrite_allowed": True,
            "evidence_allowed": True,
            "research_allowed": True,
        },
        "limits": {
            "retrieve_qps": 50,
            "retrieve_daily": 100000,
            "max_kb": 50,
            "max_kb_per_retrieve": 5,
            "max_documents_per_kb": 5000,
            "max_processing_documents": 10,
        },
        "usage": {"kb_count": 0, "retrieve_daily_count": 0},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "path", "request_kwargs"),
    [
        ("get", "/api/v1/knowledge-bases/tree", {}),
        (
            "post",
            "/api/v1/knowledge-bases/create",
            {"json": {"name": "Test KB", "description": "desc"}},
        ),
        (
            "post",
            "/api/v1/documents/upload",
            {
                "json": {
                    "kb_id": "kb-test",
                    "title": "Test",
                    "content": "content",
                }
            },
        ),
        (
            "post",
            "/api/v1/rag/retrieve",
            {
                "json": {
                    "kb_id": "kb-test",
                    "user_id": "user-test",
                    "query": "question",
                }
            },
        ),
    ],
)
async def test_business_routes_reject_missing_key(
    method: str,
    path: str,
    request_kwargs: dict,
) -> None:
    _override_auth_settings(True)
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await getattr(client, method)(path, **request_kwargs)
    finally:
        app.dependency_overrides.pop(get_settings, None)
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 401
    assert response.json()["code"] == ErrorCode.UNAUTHORIZED.code


def test_generate_api_key_returns_only_hashable_secret_metadata() -> None:
    plaintext, key_hash, key_prefix = generate_api_key()

    assert plaintext.startswith("rk_live_")
    assert len(plaintext) == len("rk_live_") + 32
    assert key_hash == hash_api_key(plaintext)
    assert key_prefix.startswith("rk_")
    assert len(key_prefix) == 7
