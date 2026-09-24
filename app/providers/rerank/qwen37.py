"""Aliyun Model Studio native text-rerank API (not the compatible reranks API)."""

import logging
import math
import time
from typing import Any

import httpx

from app.core.config import Settings
from app.providers.rerank.base import RerankProvider

LOGGER = logging.getLogger(__name__)
ENDPOINT = "/services/rerank/text-rerank/text-rerank"
INSTRUCT = (
    "Given a question, retrieve passages that contain evidence needed to answer the question."
)


class Qwen37RerankProvider(RerankProvider):
    name = "qwen3.7"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str = "qwen3.7-text-rerank",
        timeout_seconds: int = 15,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.transport = transport

    @classmethod
    def from_settings(cls, settings: Settings) -> "Qwen37RerankProvider":
        return cls(
            base_url=settings.rerank_base_url,
            api_key=settings.rerank_api_key or settings.model_api_key,
            model=settings.rerank_model,
            timeout_seconds=settings.rerank_timeout_seconds,
        )

    async def rerank(
        self,
        *,
        query: str,
        chunks: list[dict[str, Any]],
        top_n: int,
    ) -> list[dict[str, Any]]:
        candidates = chunks[:50]
        if top_n < 1 or not candidates:
            return []
        if not self.api_key or not self.base_url:
            raise ValueError("rerank credentials or endpoint not configured")
        limit = min(top_n, len(candidates))
        documents = [
            f"标题：{item.get('title') or ''}\n正文：{str(item.get('content') or '')[:1024]}"
            for item in candidates
        ]
        started = time.perf_counter()
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_seconds, transport=self.transport
            ) as client:
                response = await client.post(
                    f"{self.base_url}{ENDPOINT}",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": self.model,
                        "input": {"query": query, "documents": documents},
                        "parameters": {"top_n": limit, "instruct": INSTRUCT},
                    },
                )
                response.raise_for_status()
                payload = response.json()
        except httpx.HTTPStatusError as exc:
            raise ValueError(f"rerank HTTP {exc.response.status_code}") from None
        except httpx.TimeoutException:
            raise TimeoutError("rerank request timed out") from None
        except (httpx.HTTPError, ValueError) as exc:
            raise ValueError(f"rerank request failed ({type(exc).__name__})") from None
        finally:
            LOGGER.info(
                "rerank_request | model=%s | candidates=%s | latency_ms=%s",
                self.model,
                len(candidates),
                round((time.perf_counter() - started) * 1000),
            )

        if not isinstance(payload, dict) or not isinstance(payload.get("output"), dict):
            raise ValueError("rerank response missing output")
        results = payload["output"].get("results")
        if not isinstance(results, list) or len(results) != limit:
            raise ValueError("rerank response has invalid results count")
        seen: set[int] = set()
        ranked: list[tuple[float, int]] = []
        for item in results:
            if not isinstance(item, dict):
                raise ValueError("rerank result must be an object")
            index = item.get("index")
            if type(index) is not int or index < 0 or index >= len(candidates) or index in seen:
                raise ValueError("rerank result has invalid index")
            seen.add(index)
            if isinstance(item.get("relevance_score"), bool):
                raise ValueError("rerank result has invalid score")
            try:
                score = float(item["relevance_score"])
            except (KeyError, TypeError, ValueError, OverflowError):
                raise ValueError("rerank result has invalid score") from None
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("rerank result has invalid score")
            ranked.append((score, index))
        ranked.sort(key=lambda pair: (-pair[0], pair[1]))
        return [{**candidates[index], "rerank_score": score} for score, index in ranked]
