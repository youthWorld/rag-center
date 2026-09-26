from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any


class LLMProviderError(RuntimeError):
    """Raised when an LLM provider cannot return the expected JSON object."""


@dataclass(frozen=True, slots=True)
class LLMCallMetadata:
    provider: str | None = None
    request_model: str | None = None
    response_model: str | None = None
    latency_ms: int | None = None
    request_id: str | None = None
    finish_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LLMJSONResponse:
    output: dict[str, Any]
    metadata: LLMCallMetadata


class LLMProvider(ABC):
    @abstractmethod
    async def chat_json(
        self,
        *,
        system_prompt: str,
        user_payload: dict[str, Any],
        temperature: float = 0.0,
        timeout_seconds: float | None = None,
        log_payload: bool = True,
        max_tokens: int | None = None,
        enable_thinking: bool | None = None,
    ) -> dict[str, Any]:
        """Call a chat model and return its parsed JSON object response."""

    async def chat_json_with_metadata(
        self,
        *,
        system_prompt: str,
        user_payload: dict[str, Any],
        temperature: float = 0.0,
        timeout_seconds: float | None = None,
        log_payload: bool = True,
        max_tokens: int | None = None,
        enable_thinking: bool | None = None,
    ) -> LLMJSONResponse:
        """Return JSON plus portable call metadata for providers that support it."""

        chat_kwargs: dict[str, Any] = {
            "system_prompt": system_prompt,
            "user_payload": user_payload,
            "temperature": temperature,
            "timeout_seconds": timeout_seconds,
            "log_payload": log_payload,
            "max_tokens": max_tokens,
        }
        if enable_thinking is not None:
            chat_kwargs["enable_thinking"] = enable_thinking
        output = await self.chat_json(**chat_kwargs)
        return LLMJSONResponse(output=output, metadata=LLMCallMetadata())

    async def chat_text(
        self,
        *,
        system_prompt: str,
        user_payload: dict[str, Any],
        temperature: float = 0.0,
        timeout_seconds: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Call a chat model and return text, with a JSON fallback for older providers."""

        chat_kwargs: dict[str, Any] = {
            "system_prompt": system_prompt,
            "user_payload": user_payload,
            "temperature": temperature,
            "timeout_seconds": timeout_seconds,
        }
        if max_tokens is not None:
            chat_kwargs["max_tokens"] = max_tokens
        payload = await self.chat_json(**chat_kwargs)
        for key in ("text", "query", "rewritten_query", "content"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value
        raise LLMProviderError("chat completion returned no text field")
