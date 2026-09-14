import time

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.exceptions import KnowledgeBaseNotFoundError
from app.core.logging import get_logger
from app.providers.embedding.base import EmbeddingProvider
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
    ) -> None:
        self.session = session
        self.settings = settings
        self.knowledge_base_repository = knowledge_base_repository
        self.retrieval_log_repository = retrieval_log_repository
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
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
        retrieved = await self.vector_store.similarity_search(
            query_vector,
            tenant_id=request.tenant_id,
            kb_id=knowledge_base.id,
            top_k=self.settings.top_k,
        )
        latency_ms = int((time.perf_counter() - started_at) * 1000)

        retrieved_chunks = [
            RetrievedChunk(
                document_id=item["document_id"],
                chunk_id=item["chunk_id"],
                title=item["title"],
                content=item["content"],
                score=float(item["score"]),
            )
            for item in retrieved
        ]
        serialized_chunks = [chunk.model_dump() for chunk in retrieved_chunks]
        await self.retrieval_log_repository.create(
            tenant_id=request.tenant_id,
            kb_id=knowledge_base.id,
            user_id=request.user_id,
            query=request.query,
            retrieved_chunks=serialized_chunks,
            top_k=self.settings.top_k,
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
                "top_k": self.settings.top_k,
                "vector_store": self.settings.vector_store,
            },
        )
