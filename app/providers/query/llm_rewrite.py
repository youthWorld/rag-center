from __future__ import annotations

import asyncio
import json
import time
from dataclasses import replace
from typing import Any

from app.providers.llm.base import LLMProvider, LLMProviderError
from app.providers.query.base import QueryContext, QueryProcessor, QueryProcessResult

QUERY_REWRITE_SYSTEM_PROMPT = """你是检索 query 改写器，不是问答助手。
输入：用户原话 + 知识库名称 + 知识库领域说明。
输出：一条中文短句，更贴近文档常用表述，不超过 30 字。
不编造事实，不扩展用户没问的内容，不生成答案。
只输出改写后的句子，不输出解释。"""


class LLMRewriteProcessor(QueryProcessor):
    def __init__(
        self,
        llm_provider: LLMProvider | None,
        *,
        timeout_ms: int = 2000,
        temperature: float = 0.0,
        max_tokens: int = 128,
    ) -> None:
        if timeout_ms < 1:
            raise ValueError("timeout_ms must be at least 1")
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        self.llm_provider = llm_provider
        self.timeout_ms = timeout_ms
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def process(
        self,
        result: QueryProcessResult,
        *,
        context: QueryContext,
    ) -> QueryProcessResult:
        started_at = time.perf_counter()
        model_call_attempted = False
        try:
            if self.llm_provider is None:
                raise LLMProviderError("query rewrite LLM provider is not configured")

            description = context.description or ""
            rewrite_hint = context.settings.get("rewrite_hint")
            if isinstance(rewrite_hint, str) and rewrite_hint.strip():
                # Use description + "\n" + rewrite_hint so the hint supplements the domain context.
                description = f"{description}\n{rewrite_hint}" if description else rewrite_hint

            payload = {
                "query": result.raw_query,
                "kb_name": context.name,
                "kb_description": description,
            }
            timeout_seconds = self.timeout_ms / 1000
            model_call_attempted = True
            rewritten = await asyncio.wait_for(
                self.llm_provider.chat_text(
                    system_prompt=QUERY_REWRITE_SYSTEM_PROMPT,
                    user_payload=payload,
                    temperature=self.temperature,
                    timeout_seconds=timeout_seconds,
                    max_tokens=self.max_tokens,
                ),
                timeout=timeout_seconds,
            )
            effective_query = self._normalize_response(rewritten)
            if not effective_query:
                raise LLMProviderError("query rewrite returned empty content")
            return replace(
                result,
                effective_query=effective_query,
                search_query=effective_query,
                strategy="rewrite",
                rewrite_latency_ms=self._latency_ms(started_at),
                degraded=False,
                degraded_reason=None,
                application_model_calls=result.application_model_calls + 1,
            )
        except TimeoutError:
            return self._degraded_result(
                result,
                started_at=started_at,
                reason="query rewrite timed out",
                model_call_attempted=model_call_attempted,
            )
        except Exception as exception:
            return self._degraded_result(
                result,
                started_at=started_at,
                reason=str(exception) or type(exception).__name__,
                model_call_attempted=model_call_attempted,
            )

    @classmethod
    def _normalize_response(cls, response: Any) -> str:
        if isinstance(response, dict):
            for key in ("query", "rewritten_query", "text", "content", "rewrite"):
                value = response.get(key)
                if isinstance(value, str):
                    response = value
                    break
            else:
                return ""

        if not isinstance(response, str):
            return ""
        text = response.strip()
        if text.startswith("```") and text.endswith("```"):
            text = text[3:-3].strip()
            if text.lower().startswith("json"):
                text = text[4:].strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            return cls._normalize_response(parsed)
        if isinstance(parsed, str):
            text = parsed.strip()
        if "\n" in text:
            text = next((line.strip() for line in text.splitlines() if line.strip()), "")
        return text[:30]

    def _degraded_result(
        self,
        result: QueryProcessResult,
        *,
        started_at: float,
        reason: str,
        model_call_attempted: bool,
    ) -> QueryProcessResult:
        return replace(
            result,
            effective_query=result.raw_query,
            search_query=result.raw_query,
            strategy="rewrite",
            rewrite_latency_ms=self._latency_ms(started_at),
            degraded=True,
            degraded_reason=reason[:500],
            application_model_calls=(
                result.application_model_calls + int(model_call_attempted)
            ),
        )

    @staticmethod
    def _latency_ms(started_at: float) -> int:
        return int((time.perf_counter() - started_at) * 1000)


LLMRewriteQueryProcessor = LLMRewriteProcessor
