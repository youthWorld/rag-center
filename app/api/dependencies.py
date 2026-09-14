from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, settings
from app.db.session import get_db
from app.providers.embedding.openai_compatible import OpenAICompatibleEmbeddingProvider
from app.providers.vectorstores.pgvector import PgVectorStore
from app.repositories.document_repository import DocumentRepository
from app.repositories.knowledge_base_repository import KnowledgeBaseRepository
from app.repositories.retrieval_log_repository import RetrievalLogRepository
from app.services.document_service import DocumentService
from app.services.indexing_service import IndexingService
from app.services.knowledge_base_service import KnowledgeBaseService
from app.services.rag_service import RagService
from app.utils.text_splitter import TextSplitter


def get_settings() -> Settings:
    return settings


def get_knowledge_base_service(
    session: AsyncSession = Depends(get_db),
) -> KnowledgeBaseService:
    return KnowledgeBaseService(
        session=session,
        repository=KnowledgeBaseRepository(session),
    )


def get_document_service(
    session: AsyncSession = Depends(get_db),
    app_settings: Settings = Depends(get_settings),
) -> DocumentService:
    embedding_provider = OpenAICompatibleEmbeddingProvider(app_settings)
    vector_store = PgVectorStore(session)
    indexing_service = IndexingService(
        splitter=TextSplitter(
            chunk_size=app_settings.chunk_size,
            chunk_overlap=app_settings.chunk_overlap,
        ),
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )
    return DocumentService(
        session=session,
        document_repository=DocumentRepository(session),
        knowledge_base_repository=KnowledgeBaseRepository(session),
        indexing_service=indexing_service,
    )


def get_rag_service(
    session: AsyncSession = Depends(get_db),
    app_settings: Settings = Depends(get_settings),
) -> RagService:
    return RagService(
        session=session,
        settings=app_settings,
        knowledge_base_repository=KnowledgeBaseRepository(session),
        retrieval_log_repository=RetrievalLogRepository(session),
        embedding_provider=OpenAICompatibleEmbeddingProvider(app_settings),
        vector_store=PgVectorStore(session),
    )
