from typing import Any

from app.providers.rerank.base import RerankProvider


class NoopRerankProvider(RerankProvider):
    async def rerank(
        self,
        *,
        query: str,
        chunks: list[dict[str, Any]],
        top_n: int,
    ) -> list[dict[str, Any]]:
        del query
        return [
            {**chunk, "rerank_score": None}
            for chunk in chunks[: max(top_n, 0)]
        ]
