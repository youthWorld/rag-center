from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from app.providers.query.base import (
    QueryContext,
    QueryProcessor,
    QueryProcessResult,
)
from app.providers.query.noop import NoopQueryProcessor
from app.providers.query.synonym_expander import SynonymExpander


class QueryPipeline:
    def __init__(
        self,
        *,
        rewrite_enabled: bool = False,
        rewrite_processor: QueryProcessor | None = None,
        synonym_processor: QueryProcessor | None = None,
    ) -> None:
        self.rewrite_enabled = rewrite_enabled
        self.rewrite_processor = rewrite_processor
        self.noop_processor = NoopQueryProcessor()
        self.synonym_processor = synonym_processor or SynonymExpander()

    async def process(
        self,
        raw_query: str,
        *,
        knowledge_base: Any,
        query_options: Any = None,
    ) -> QueryProcessResult:
        context = QueryContext.from_knowledge_base(knowledge_base)
        rewrite_enabled, strategy = self._resolve_strategy(query_options)
        result = QueryProcessResult(
            raw_query=raw_query,
            effective_query=raw_query,
            search_query=raw_query,
            strategy=strategy,
        )

        if rewrite_enabled and strategy == "rewrite":
            if self.rewrite_processor is None:
                result = replace(
                    result,
                    degraded=True,
                    degraded_reason="query rewrite processor is not configured",
                )
            else:
                try:
                    result = await self.rewrite_processor.process(result, context=context)
                except Exception as exception:
                    result = replace(
                        result,
                        degraded=True,
                        degraded_reason=str(exception) or type(exception).__name__,
                    )
        else:
            result = await self.noop_processor.process(result, context=context)

        return await self.synonym_processor.process(result, context=context)

    def _resolve_strategy(self, query_options: Any) -> tuple[bool, str]:
        if query_options is None:
            enabled = self.rewrite_enabled
            return enabled, "rewrite" if enabled else "noop"

        if isinstance(query_options, Mapping):
            requested_enabled = query_options.get("enabled")
            requested_strategy = query_options.get("strategy")
        else:
            requested_enabled = getattr(query_options, "enabled", None)
            requested_strategy = getattr(query_options, "strategy", None)
        if requested_strategy == "noop":
            return False, "noop"
        if requested_enabled is not None:
            enabled = bool(requested_enabled)
        elif requested_strategy == "rewrite":
            enabled = True
        else:
            enabled = self.rewrite_enabled
        return enabled, "rewrite" if enabled else "noop"
