import math
from typing import Any

from app.core.config import Settings
from app.providers.llm.base import LLMProvider
from app.providers.rerank.base import RerankProvider
from app.schemas.rerank import RerankCandidate

RERANK_SYSTEM_PROMPT = """你是 RAG 检索重排器。你的任务是根据用户问题，
对候选知识片段进行相关性打分和排序。

要求：
1. 只判断候选片段是否有助于回答用户问题。
2. 不要回答用户问题。
3. 不要编造候选片段中不存在的信息。
4. 每个候选片段给出 0 到 1 之间的 rerank_score。
5. 分数越高表示越相关、越适合作为 RAG 上下文。
6. 只返回 JSON，不要返回 Markdown，不要返回解释性文字。

返回格式：
{"rankings":[{"chunk_id":"...","rerank_score":0.0,"reason":"..."}]}
"""


class LLMRerankProvider(RerankProvider):
    def __init__(
        self,
        llm_provider: LLMProvider,
        *,
        max_candidates: int = 20,
        chunk_max_chars: int = 1000,
        temperature: float = 0.0,
        timeout_seconds: int | None = None,
    ) -> None:
        if max_candidates < 1:
            raise ValueError("max_candidates must be at least 1")
        if chunk_max_chars < 1:
            raise ValueError("chunk_max_chars must be at least 1")
        self.llm_provider = llm_provider
        self.max_candidates = max_candidates
        self.chunk_max_chars = chunk_max_chars
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds

    @classmethod
    def from_settings(cls, llm_provider: LLMProvider, settings: Settings) -> "LLMRerankProvider":
        return cls(
            llm_provider,
            max_candidates=settings.rerank_max_candidates,
            chunk_max_chars=settings.rerank_chunk_max_chars,
            temperature=settings.rerank_temperature,
            timeout_seconds=settings.llm_timeout_seconds,
        )

    def _build_payload(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        top_n: int,
    ) -> dict[str, Any]:
        candidates = [
            RerankCandidate(
                chunk_id=str(chunk["chunk_id"]),
                document_id=str(chunk["document_id"]),
                title=str(chunk.get("title", "")),
                content=str(chunk.get("content", ""))[: self.chunk_max_chars],
                vector_score=float(chunk.get("score", 0.0)),
            ).model_dump()
            for chunk in chunks[: self.max_candidates]
        ]
        return {"query": query, "candidates": candidates, "top_n": top_n}

    async def rerank(
        self,
        *,
        query: str,
        chunks: list[dict[str, Any]],
        top_n: int,
    ) -> list[dict[str, Any]]:
        if top_n < 1:
            return []
        candidates = chunks[: self.max_candidates]
        if not candidates:
            return []

        payload = self._build_payload(query, candidates, top_n)
        response = await self.llm_provider.chat_json(
            system_prompt=RERANK_SYSTEM_PROMPT,
            user_payload=payload,
            temperature=self.temperature,
            timeout_seconds=self.timeout_seconds,
        )
        rankings = response.get("rankings")
        if not isinstance(rankings, list):
            raise ValueError("rerank response must contain a rankings array")

        candidate_ids = {str(chunk["chunk_id"]) for chunk in candidates}
        scores: dict[str, float] = {}
        for ranking in rankings:
            if not isinstance(ranking, dict):
                continue
            chunk_id = ranking.get("chunk_id")
            if chunk_id is None or str(chunk_id) not in candidate_ids:
                continue
            try:
                score = float(ranking.get("rerank_score", 0.0))
            except (TypeError, ValueError):
                score = 0.0
            if not math.isfinite(score):
                score = 0.0
            scores.setdefault(str(chunk_id), min(max(score, 0.0), 1.0))

        ranked_chunks = [
            {
                **chunk,
                "rerank_score": scores.get(str(chunk["chunk_id"]), 0.0),
            }
            for chunk in candidates
        ]
        ranked_chunks.sort(
            key=lambda chunk: (
                -float(chunk["rerank_score"]),
                -float(chunk.get("score", 0.0)),
            )
        )
        return ranked_chunks[:top_n]
