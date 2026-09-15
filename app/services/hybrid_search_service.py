from typing import Any

from app.schemas.hybrid_search import HybridSearchChunk


class HybridSearchService:
    """Normalize and fuse vector/BM25 results with reciprocal rank fusion."""

    def __init__(self, *, rrf_k: int = 60) -> None:
        if rrf_k < 1:
            raise ValueError("rrf_k must be at least 1")
        self.rrf_k = rrf_k

    def normalize_vector_results(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for rank, chunk in enumerate(chunks, start=1):
            score = self._score(chunk.get("vector_score", chunk.get("score")))
            normalized.append(
                {
                    **chunk,
                    "score": score,
                    "vector_score": score,
                    "bm25_score": None,
                    "vector_rank": rank,
                    "bm25_rank": None,
                    "retrieval_source": "vector",
                }
            )
        return normalized

    def normalize_bm25_results(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for rank, chunk in enumerate(chunks, start=1):
            score = self._score(chunk.get("bm25_score", chunk.get("score")))
            normalized.append(
                {
                    **chunk,
                    "score": score,
                    "vector_score": None,
                    "bm25_score": score,
                    "vector_rank": None,
                    "bm25_rank": rank,
                    "retrieval_source": "bm25",
                }
            )
        return normalized

    def fuse(
        self,
        vector_chunks: list[dict[str, Any]],
        bm25_chunks: list[dict[str, Any]],
        *,
        rrf_k: int | None = None,
        top_n: int | None = None,
    ) -> list[dict[str, Any]]:
        effective_rrf_k = rrf_k if rrf_k is not None else self.rrf_k
        if effective_rrf_k < 1:
            raise ValueError("rrf_k must be at least 1")
        if top_n is not None and top_n < 0:
            raise ValueError("top_n must not be negative")

        merged: dict[str, dict[str, Any]] = {}
        self._merge_source(merged, vector_chunks, source="vector")
        self._merge_source(merged, bm25_chunks, source="bm25")

        fused: list[dict[str, Any]] = []
        for chunk in merged.values():
            vector_rank = chunk["vector_rank"]
            bm25_rank = chunk["bm25_rank"]
            fused_score = 0.0
            if vector_rank is not None:
                fused_score += 1 / (effective_rrf_k + vector_rank)
            if bm25_rank is not None:
                fused_score += 1 / (effective_rrf_k + bm25_rank)

            if vector_rank is not None and bm25_rank is not None:
                source = "hybrid"
            elif vector_rank is not None:
                source = "vector"
            else:
                source = "bm25"

            fused.append(
                HybridSearchChunk(
                    document_id=chunk["document_id"],
                    chunk_id=chunk["chunk_id"],
                    title=chunk["title"],
                    content=chunk["content"],
                    score=fused_score,
                    vector_score=chunk["vector_score"],
                    bm25_score=chunk["bm25_score"],
                    vector_rank=vector_rank,
                    bm25_rank=bm25_rank,
                    retrieval_source=source,
                ).model_dump()
            )

        fused.sort(key=self._sort_key)
        return fused if top_n is None else fused[:top_n]

    def _merge_source(
        self,
        merged: dict[str, dict[str, Any]],
        chunks: list[dict[str, Any]],
        *,
        source: str,
    ) -> None:
        for rank, chunk in enumerate(chunks, start=1):
            chunk_id = chunk.get("chunk_id") or chunk.get("id")
            if chunk_id is None:
                continue
            chunk_id = str(chunk_id)
            record = merged.setdefault(
                chunk_id,
                {
                    "document_id": str(chunk.get("document_id", "")),
                    "chunk_id": chunk_id,
                    "title": str(chunk.get("title", "")),
                    "content": str(chunk.get("content", "")),
                    "vector_score": None,
                    "bm25_score": None,
                    "vector_rank": None,
                    "bm25_rank": None,
                },
            )
            if not record["document_id"] and chunk.get("document_id") is not None:
                record["document_id"] = str(chunk["document_id"])
            if not record["title"] and chunk.get("title") is not None:
                record["title"] = str(chunk["title"])
            if not record["content"] and chunk.get("content") is not None:
                record["content"] = str(chunk["content"])

            if source == "vector" and record["vector_rank"] is None:
                record["vector_rank"] = rank
                record["vector_score"] = self._score(
                    chunk.get("vector_score", chunk.get("score"))
                )
            elif source == "bm25" and record["bm25_rank"] is None:
                record["bm25_rank"] = rank
                record["bm25_score"] = self._score(
                    chunk.get("bm25_score", chunk.get("score"))
                )

    @staticmethod
    def _score(value: Any) -> float:
        return float(value) if value is not None else 0.0

    @staticmethod
    def _sort_key(chunk: dict[str, Any]) -> tuple[float, int, int, str]:
        return (
            -float(chunk["score"]),
            chunk["vector_rank"] or 10**9,
            chunk["bm25_rank"] or 10**9,
            chunk["chunk_id"],
        )
