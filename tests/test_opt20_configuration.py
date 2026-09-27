from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from app.core import http_clients
from app.core.config import Settings


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    for key in (
        "HYBRID_RRF_K",
        "MODEL_HTTP_MAX_CONNECTIONS",
        "MODEL_HTTP_MAX_KEEPALIVE_CONNECTIONS",
        "MODEL_HTTP_KEEPALIVE_EXPIRY_SECONDS",
    ):
        monkeypatch.delenv(key, raising=False)


def test_rrf_configuration_is_required():
    with pytest.raises(ValidationError) as raised:
        Settings(_env_file=None)
    assert any(error["loc"] == ("hybrid_rrf_k",) for error in raised.value.errors())


@pytest.mark.parametrize("value", ["0", "-1", "invalid", "1.5", ""])
def test_invalid_rrf_environment_is_rejected(monkeypatch, value):
    monkeypatch.setenv("HYBRID_RRF_K", value)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_pool_configuration_from_environment(monkeypatch):
    monkeypatch.setenv("HYBRID_RRF_K", "71")
    monkeypatch.setenv("MODEL_HTTP_MAX_CONNECTIONS", "40")
    monkeypatch.setenv("MODEL_HTTP_MAX_KEEPALIVE_CONNECTIONS", "12")
    monkeypatch.setenv("MODEL_HTTP_KEEPALIVE_EXPIRY_SECONDS", "15.5")
    config = Settings(_env_file=None)
    assert config.hybrid_rrf_k == 71
    assert config.model_http_max_connections == 40
    assert config.model_http_max_keepalive_connections == 12
    assert config.model_http_keepalive_expiry_seconds == 15.5


@pytest.mark.parametrize(
    "overrides",
    [
        {"model_http_max_connections": 0},
        {"model_http_max_keepalive_connections": -1},
        {"model_http_max_keepalive_connections": 101},
        {"model_http_keepalive_expiry_seconds": 0},
        {"model_http_keepalive_expiry_seconds": float("inf")},
        {"model_http_keepalive_expiry_seconds": float("nan")},
    ],
)
def test_invalid_pool_configuration_is_rejected(overrides):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, hybrid_rrf_k=60, **overrides)


@pytest.mark.asyncio
async def test_container_uses_configured_limits_without_changing_es_defaults(monkeypatch):
    seen = {}
    model = SimpleNamespace(aclose=AsyncMock())
    es = SimpleNamespace(close=AsyncMock())

    def model_factory(**kwargs):
        seen["model"] = kwargs
        return model

    def es_factory(*args, **kwargs):
        seen["es"] = (args, kwargs)
        return es

    monkeypatch.setattr(http_clients.httpx, "AsyncClient", model_factory)
    monkeypatch.setattr(http_clients, "AsyncElasticsearch", es_factory)
    settings = Settings(
        _env_file=None,
        hybrid_rrf_k=60,
        model_http_max_connections=40,
        model_http_max_keepalive_connections=12,
        model_http_keepalive_expiry_seconds=15.5,
    )
    clients = http_clients.build_external_clients(settings)
    limits = seen["model"]["limits"]
    assert limits.max_connections == 40
    assert limits.max_keepalive_connections == 12
    assert limits.keepalive_expiry == 15.5
    assert seen["es"] == ((settings.elasticsearch_url,), {})
    assert "headers" not in seen["model"]
    await clients.aclose()
    model.aclose.assert_awaited_once()
    es.close.assert_awaited_once()
