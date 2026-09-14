import time
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.exceptions import KnowledgeBaseNotFoundError
from app.core.logging import get_logger
from app.providers.embedding.base import EmbeddingProvider
from app.providers.rerank.base import RerankProvider
from app.providers.rerank.noop import NoopRerankProvider
from app.providers.vectorstores.base import VectorStore
from app.repositories.knowledge_base_repository import KnowledgeBaseRepository
from app.repositories.retrieval_log_repository import RetrievalLogRepository
from app.schemas.rag import RagRetrieveRequest, RagRetrieveResponse, RetrievedChunk


class RagService:
    def __init__(
        self,
        session: AsyncSession,
        settings: Settings,
        knowledge_base_repository: KnowledgeBaseRepository,
        retrieval_log_repository: RetrievalLogRepository,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        rerank_provider: RerankProvider | None = None,
    ) -> None:
        self.session = session
        self.settings = settings
        self.knowledge_base_repository = knowledge_base_repository
        self.retrieval_log_repository = retrieval_log_repository
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.rerank_provider = rerank_provider or NoopRerankProvider()
        self.logger = get_logger(__name__)

    async def retrieve(self, request: RagRetrieveRequest) -> RagRetrieveResponse:
        self.logger.info(
            "BUSINESS_EVENT | event=rag_retrieval_started | kb_id=%s | tenant_id=%s | user_id=%s",
            request.kb_id,
            request.tenant_id,
            request.user_id,
        )
        knowledge_base = await self.knowledge_base_repository.get_by_id(
            kb_id=request.kb_id,
            tenant_id=request.tenant_id,
        )
        if knowledge_base is None:
            raise KnowledgeBaseNotFoundError()

        started_at = time.perf_counter()
        query_vector = await self.embedding_provider.embed_query(request.query)
        top_k = request.top_k or self.settings.top_k
        retrieved = await self.vector_store.similarity_search(
            query_vector,
            tenant_id=request.tenant_id,
            kb_id=knowledge_base.id,
            top_k=top_k,
        )

        rerank_enabled = self._resolve_rerank_enabled(request)
        rerank_top_n = self._resolve_rerank_top_n(request)
        candidate_count = 0
        degraded = False
        rerank_error: str | None = None
        reranked = retrieved
        if rerank_enabled:
            rerank_candidates = retrieved[: self.settings.rerank_max_candidates]
            candidate_count = len(rerank_candidates)
            try:
                reranked = await self.rerank_provider.rerank(
                    query=request.query,
                    chunks=rerank_candidates,
                    top_n=rerank_top_n,
                )
            except Exception as exception:
                degraded = True
                rerank_error = str(exception) or type(exception).__name__
                reranked = retrieved
                self.logger.exception(
                    "BUSINESS_EVENT | event=rag_rerank_degraded | kb_id=%s | "
                    "candidate_count=%s | error=%s",
                    knowledge_base.id,
                    candidate_count,
                    rerank_error,
                )

        latency_ms = int((time.perf_counter() - started_at) * 1000)
        response_chunks = (
            reranked[:rerank_top_n]
            if rerank_enabled and not degraded
            else reranked
        )
        retrieved_chunks = [
            RetrievedChunk(
                document_id=item["document_id"],
                chunk_id=item["chunk_id"],
                title=item["title"],
                content=item["content"],
                score=float(item["score"]),
                rerank_score=(
                    float(item["rerank_score"])
                    if item.get("rerank_score") is not None
                    else None
                ),
            )
            for item in response_chunks
        ]
        serialized_chunks = [chunk.model_dump() for chunk in retrieved_chunks]
        await self.retrieval_log_repository.create(
            tenant_id=request.tenant_id,
            kb_id=knowledge_base.id,
            user_id=request.user_id,
            query=request.query,
            retrieved_chunks=serialized_chunks,
            top_k=top_k,
            vector_store=self.settings.vector_store,
            latency_ms=latency_ms,
        )
        await self.session.commit()
        self.logger.info(
            "BUSINESS_EVENT | event=rag_retrieval_completed | kb_id=%s | "
            "chunk_count=%s | latency_ms=%s",
            knowledge_base.id,
            len(retrieved_chunks),
            latency_ms,
        )

        return RagRetrieveResponse(
            query=request.query,
            kb_id=knowledge_base.id,
            retrieved_chunks=retrieved_chunks,
            metadata={
                "top_k": top_k,
                "vector_store": self.settings.vector_store,
                "rerank": self._build_rerank_metadata(
                    enabled=rerank_enabled,
                    top_n=rerank_top_n,
                    candidate_count=candidate_count,
                    degraded=degraded,
                    error=rerank_error,
                ),
            },
        )

    def _resolve_rerank_enabled(self, request: RagRetrieveRequest) -> bool:
        if request.rerank_options is not None and request.rerank_options.enabled is not None:
            return request.rerank_options.enabled
        return self.settings.rerank_enabled

    def _resolve_rerank_top_n(self, request: RagRetrieveRequest) -> int:
        if request.rerank_options is not None and request.rerank_options.top_n is not None:
            return request.rerank_options.top_n
        return self.settings.rerank_top_n

    def _build_rerank_metadata(
        self,
        *,
        enabled: bool,
        top_n: int,
        candidate_count: int,
        degraded: bool,
        error: str | None,
    ) -> dict[str, Any]:
        provider = self.settings.rerank_provider if enabled else "noop"
        metadata: dict[str, Any] = {
            "enabled": enabled,
            "provider": provider,
            "llm_provider": self.settings.llm_provider if provider == "llm" else None,
            "model": self.settings.llm_model if provider == "llm" else None,
            "top_n": top_n,
            "candidate_count": candidate_count,
        }
        if degraded:
            metadata["degraded"] = True
            metadata["error"] = error or "rerank failed"
        return metadata
