from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

QueryStrategy = Literal["noop", "rewrite"]


@dataclass(frozen=True)
class QueryContext:
    """Knowledge-base context available to query processors."""

    name: str
    description: str | None = None
    settings: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_knowledge_base(cls, knowledge_base: Any) -> QueryContext:
        raw_settings = getattr(knowledge_base, "settings", {})
        settings = raw_settings if isinstance(raw_settings, Mapping) else {}
        return cls(
            name=str(getattr(knowledge_base, "name", "") or ""),
            description=(
                str(getattr(knowledge_base, "description", ""))
                if getattr(knowledge_base, "description", None) is not None
                else None
            ),
            settings=settings,
        )


@dataclass
class QueryProcessResult:
    raw_query: str
    effective_query: str
    search_query: str
    strategy: QueryStrategy = "noop"
    rewrite_latency_ms: int = 0
    degraded: bool = False
    degraded_reason: str | None = None
    synonym_applied: bool = False
    synonym_expansions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def should_expose(self) -> bool:
        return (
            self.strategy == "rewrite"
            or self.degraded
            or self.effective_query != self.raw_query
            or self.search_query != self.raw_query
            or self.synonym_applied
        )


class QueryProcessor(ABC):
    @abstractmethod
    async def process(
        self,
        result: QueryProcessResult,
        *,
        context: QueryContext,
    ) -> QueryProcessResult:
        """Process a query stage while preserving the accumulated result."""
