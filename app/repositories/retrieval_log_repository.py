from typing import Any

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.retrieval_log import RetrievalLog
from app.utils.id_generator import generate_id


class RetrievalLogRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self,
        *,
        tenant_id: str,
        kb_id: str,
        user_id: str,
        query: str,
        trace_id: str | None = None,
        profile: str | None = None,
        search_query: str | None = None,
        effective_query: str | None = None,
        retrieved_chunks: list[dict[str, Any]],
        top_k: int,
        vector_store: str,
        latency_ms: int,
    ) -> RetrievalLog:
        log = RetrievalLog(
            id=generate_id(),
            tenant_id=tenant_id,
            kb_id=kb_id,
            user_id=user_id,
            query=query,
            trace_id=trace_id,
            profile=profile,
            search_query=search_query,
            effective_query=effective_query,
            retrieved_chunks=retrieved_chunks,
            top_k=top_k,
            vector_store=vector_store,
            latency_ms=latency_ms,
        )
        self.session.add(log)
        await self.session.flush()
        return log

    async def get_by_id(self, *, log_id: str) -> RetrievalLog | None:
        statement = select(RetrievalLog).where(RetrievalLog.id == log_id)
        result = await self.session.execute(statement)
        return result.scalar_one_or_none()

    async def acquire_feedback_lock(self, *, trace_id: str) -> None:
        await self.session.execute(
            text("SELECT pg_advisory_xact_lock(hashtextextended(:trace_id, 0))"),
            {"trace_id": trace_id},
        )
