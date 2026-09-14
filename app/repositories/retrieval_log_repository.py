from typing import Any

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
            retrieved_chunks=retrieved_chunks,
            top_k=top_k,
            vector_store=vector_store,
            latency_ms=latency_ms,
        )
        self.session.add(log)
        await self.session.flush()
        return log
