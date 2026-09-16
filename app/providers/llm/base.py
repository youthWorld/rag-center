from abc import ABC, abstractmethod
from typing import Any


class LLMProviderError(RuntimeError):
    """Raised when an LLM provider cannot return the expected JSON object."""


class LLMProvider(ABC):
    @abstractmethod
    async def chat_json(
        self,
        *,
        system_prompt: str,
        user_payload: dict[str, Any],
        temperature: float = 0.0,
        timeout_seconds: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Call a chat model and return its parsed JSON object response."""

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
