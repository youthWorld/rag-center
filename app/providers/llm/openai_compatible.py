import json
from typing import Any

from openai import AsyncOpenAI

from app.core.config import Settings
from app.core.exceptions import (
    LLMServiceError,
    ServiceConfigurationError,
    map_llm_exception,
)
from app.core.logging import get_logger, log_llm_call
from app.providers.llm.base import LLMProvider, LLMProviderError


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
    ) -> dict[str, Any]:
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
        return payload

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
