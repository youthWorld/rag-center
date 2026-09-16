from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from app.providers.query.base import QueryContext, QueryProcessor, QueryProcessResult


class SynonymExpander(QueryProcessor):
    async def process(
        self,
        result: QueryProcessResult,
        *,
        context: QueryContext,
    ) -> QueryProcessResult:
        search_query, applied, expansions = self.expand(
            result.effective_query,
            context.settings,
        )
        if not applied:
            return result
        return replace(
            result,
            search_query=search_query,
            synonym_applied=True,
            synonym_expansions=expansions,
        )

    @staticmethod
    def expand(
        query: str,
        settings: Mapping[str, Any] | None,
    ) -> tuple[str, bool, list[str]]:
        groups = settings.get("synonyms") if isinstance(settings, Mapping) else None
        if not isinstance(groups, list):
            return query, False, []

        normalized_query = query.casefold()
        expansions: list[str] = []
        expansion_keys: set[str] = set()
        applied = False
        for group in groups:
            if not isinstance(group, Mapping):
                continue
            terms = group.get("terms")
            expand = group.get("expand")
            if not isinstance(terms, list) or not isinstance(expand, list):
                continue
            if not any(
                isinstance(term, str)
                and term.strip()
                and term.strip().casefold() in normalized_query
                for term in terms
            ):
                continue

            applied = True
            for value in expand:
                if not isinstance(value, str):
                    continue
                normalized_value = value.strip()
                if not normalized_value:
                    continue
                key = normalized_value.casefold()
                if key in expansion_keys:
                    continue
                expansion_keys.add(key)
                expansions.append(normalized_value)

        if not expansions:
            return query, applied, []
        return f"{query} {' '.join(expansions)}", applied, expansions
