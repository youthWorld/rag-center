from __future__ import annotations

from enum import StrEnum

from app.core.config import Settings
from app.providers.llm.openai_compatible import OpenAICompatibleLLMProvider


class LLMRole(StrEnum):
    FAST = "fast"
    STRONG = "strong"


class LLMRoleRouter:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def model_for(self, role: LLMRole) -> str:
        if role == LLMRole.STRONG:
            return self.settings.llm_strong_model.strip() or self.settings.llm_model
        return self.settings.llm_model

    def provider_for(self, role: LLMRole) -> OpenAICompatibleLLMProvider:
        return OpenAICompatibleLLMProvider(
            self.settings,
            model_override=self.model_for(role),
        )
