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
        timeout_seconds: int | None = None,
    ) -> dict[str, Any]:
        """Call a chat model and return its parsed JSON object response."""
