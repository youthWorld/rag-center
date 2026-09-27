from __future__ import annotations

import asyncio
from dataclasses import dataclass

import httpx
from elasticsearch import AsyncElasticsearch
from openai import AsyncOpenAI

from app.core.config import Settings


@dataclass(slots=True)
class ExternalClients:
    model: httpx.AsyncClient
    elasticsearch: AsyncElasticsearch

    _closed: bool = False

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        await asyncio.gather(self.model.aclose(), self.elasticsearch.close())


def build_external_clients(settings: Settings) -> ExternalClients:
    limits = httpx.Limits(
        max_connections=settings.model_http_max_connections,
        max_keepalive_connections=settings.model_http_max_keepalive_connections,
        keepalive_expiry=settings.model_http_keepalive_expiry_seconds,
    )
    return ExternalClients(
        model=httpx.AsyncClient(limits=limits, trust_env=True, follow_redirects=True),
        elasticsearch=AsyncElasticsearch(settings.elasticsearch_url),
    )


class BorrowedAsyncOpenAI(AsyncOpenAI):
    """SDK wrapper borrowing a transport owned by the application or caller."""

    async def close(self) -> None:
        pass
