import json
import time
from typing import Any

from openai import AsyncOpenAI

from app.core.config import Settings
from app.core.exceptions import (
    LLMServiceError,
    ServiceConfigurationError,
    map_llm_exception,
)
from app.core.logging import get_logger, log_llm_call
from app.providers.llm.base import (
    LLMCallMetadata,
    LLMJSONResponse,
    LLMProvider,
    LLMProviderError,
)


class OpenAICompatibleLLMProvider(LLMProvider):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: AsyncOpenAI | None = None
        self.logger = get_logger(__name__)

    def _get_client(self) -> AsyncOpenAI:
        if not self.settings.llm_api_key:
            raise ServiceConfigurationError(
                internal_message="LLM_API_KEY is not configured",
                context={"provider": type(self).__name__},
            )
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self.settings.llm_api_key,
                base_url=self.settings.llm_base_url,
            )
        return self._client

    @staticmethod
    def _extract_content(response: Any) -> str:
        choices = (
            response.get("choices")
            if isinstance(response, dict)
            else getattr(response, "choices", None)
        ) or []
        if not choices:
            raise LLMProviderError("chat completion returned no choices")

        choice = choices[0]
        message = (
            choice.get("message")
            if isinstance(choice, dict)
            else getattr(choice, "message", None)
        )
        content = (
            message.get("content")
            if isinstance(message, dict)
            else getattr(message, "content", None)
        )
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                    continue
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    parts.append(text)
                    continue
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            rendered = "".join(parts)
            if rendered.strip():
                return rendered
        raise LLMProviderError("chat completion returned empty content")

    async def chat_json(
        self,
        *,
        system_prompt: str,
        user_payload: dict[str, Any],
        temperature: float = 0.0,
        timeout_seconds: float | None = None,
        max_tokens: int | None = None,
        enable_thinking: bool | None = None,
    ) -> dict[str, Any]:
        result = await self.chat_json_with_metadata(
            system_prompt=system_prompt,
            user_payload=user_payload,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
        )
        return result.output

    async def chat_json_with_metadata(
        self,
        *,
        system_prompt: str,
        user_payload: dict[str, Any],
        temperature: float = 0.0,
        timeout_seconds: float | None = None,
        max_tokens: int | None = None,
        enable_thinking: bool | None = None,
    ) -> LLMJSONResponse:
        request_kwargs: dict[str, Any] = {
            "model": self.settings.llm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(user_payload, ensure_ascii=False),
                },
            ],
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }
        if max_tokens is not None:
            request_kwargs["max_tokens"] = max_tokens
        if timeout_seconds is not None:
            request_kwargs["timeout"] = timeout_seconds
        if enable_thinking is not None and self._supports_enable_thinking():
            request_kwargs["extra_body"] = {"enable_thinking": enable_thinking}

        started = time.perf_counter()
        response = await self._chat_completion(
            request_kwargs=request_kwargs,
            user_payload=user_payload,
            operation="chat_json",
        )
        content = self._extract_content(response)
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exception:
            raise LLMProviderError("chat completion returned invalid JSON") from exception
        if not isinstance(payload, dict):
            raise LLMProviderError("chat completion JSON response must be an object")
        latency_ms = int((time.perf_counter() - started) * 1000)
        return LLMJSONResponse(
            output=payload,
            metadata=LLMCallMetadata(
                provider=self.settings.llm_provider,
                request_model=self.settings.llm_model,
                response_model=self._read_string(response, "model"),
                latency_ms=latency_ms,
                request_id=(
                    self._read_string(response, "request_id")
                    or self._read_string(response, "_request_id")
                    or self._read_string(response, "id")
                ),
                finish_reason=self._finish_reason(response),
                input_tokens=self._usage_value(response, "prompt_tokens", "input_tokens"),
                output_tokens=self._usage_value(
                    response, "completion_tokens", "output_tokens"
                ),
                total_tokens=self._usage_value(response, "total_tokens"),
            ),
        )

    def _supports_enable_thinking(self) -> bool:
        model = self.settings.llm_model.lower()
        base_url = self.settings.llm_base_url.lower()
        return "dashscope.aliyuncs.com" in base_url and model.startswith(
            ("qwen3", "qwen-plus", "qwen-flash")
        )

    @staticmethod
    def _read_value(value: Any, key: str) -> Any:
        return value.get(key) if isinstance(value, dict) else getattr(value, key, None)

    @classmethod
    def _read_string(cls, value: Any, key: str) -> str | None:
        item = cls._read_value(value, key)
        return str(item) if item is not None and str(item).strip() else None

    @classmethod
    def _finish_reason(cls, response: Any) -> str | None:
        choices = cls._read_value(response, "choices") or []
        if not choices:
            return None
        return cls._read_string(choices[0], "finish_reason")

    @classmethod
    def _usage_value(cls, response: Any, *keys: str) -> int | None:
        usage = cls._read_value(response, "usage")
        for key in keys:
            value = cls._read_value(usage, key) if usage is not None else None
            if isinstance(value, int) and not isinstance(value, bool):
                return value
        return None

    async def chat_text(
        self,
        *,
        system_prompt: str,
        user_payload: dict[str, Any],
        temperature: float = 0.0,
        timeout_seconds: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        request_kwargs: dict[str, Any] = {
            "model": self.settings.llm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(user_payload, ensure_ascii=False),
                },
            ],
            "temperature": temperature,
        }
        if max_tokens is not None:
            request_kwargs["max_tokens"] = max_tokens
        if timeout_seconds is not None:
            request_kwargs["timeout"] = timeout_seconds

        response = await self._chat_completion(
            request_kwargs=request_kwargs,
            user_payload=user_payload,
            operation="chat_text",
        )
        return self._extract_content(response).strip()

    async def _chat_completion(
        self,
        *,
        request_kwargs: dict[str, Any],
        user_payload: dict[str, Any],
        operation: str,
    ) -> Any:
        try:
            response = await log_llm_call(
                lambda: self._get_client().chat.completions.create(**request_kwargs),
                model=self.settings.llm_model,
                prompt=user_payload,
                logger=self.logger,
                response_formatter=lambda value: {"choices": len(getattr(value, "choices", []))},
            )
        except (ServiceConfigurationError, LLMProviderError):
            raise
        except LLMServiceError:
            raise
        except Exception as exception:
            raise map_llm_exception(
                exception,
                model=self.settings.llm_model,
                operation=operation,
            ) from exception
        return response
