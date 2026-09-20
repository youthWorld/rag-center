from collections.abc import Mapping, Sequence
from typing import Any


class MultiKBFusionService:
    """Fuse per-knowledge-base candidates using reciprocal rank fusion."""

    def __init__(self, *, rrf_k: int = 60) -> None:
        if rrf_k < 1:
            raise ValueError("rrf_k must be at least 1")
        self.rrf_k = rrf_k

    def fuse(
        self,
        per_kb_chunks: Mapping[str, Sequence[dict[str, Any]]],
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
        for kb_id, chunks in per_kb_chunks.items():
            for rank, chunk in enumerate(chunks, start=1):
                chunk_id = chunk.get("chunk_id") or chunk.get("id")
                if chunk_id is None:
                    continue
                chunk_id = str(chunk_id)
                record = merged.get(chunk_id)
                if record is None:
                    record = {
                        **chunk,
                        "chunk_id": chunk_id,
                        "score": 0.0,
                        "kb_id": str(chunk.get("kb_id") or kb_id),
                        "kb_name": chunk.get("kb_name"),
                        "metadata": dict(chunk.get("metadata") or {}),
                    }
                    merged[chunk_id] = record
                elif record.get("kb_name") is None and chunk.get("kb_name") is not None:
                    record["kb_name"] = chunk["kb_name"]

                record["score"] += 1 / (effective_rrf_k + rank)

        fused: list[dict[str, Any]] = []
        for record in merged.values():
            fused.append(record)

        fused.sort(
            key=lambda chunk: (
                -float(chunk["score"]),
                str(chunk.get("chunk_id", "")),
            )
        )
        return fused if top_n is None else fused[:top_n]

    def fuse_by_rrf(
        self,
        per_kb_chunks: Mapping[str, Sequence[dict[str, Any]]],
        *,
        rrf_k: int | None = None,
        top_n: int | None = None,
    ) -> list[dict[str, Any]]:
        return self.fuse(per_kb_chunks, rrf_k=rrf_k, top_n=top_n)
