from collections.abc import AsyncIterator

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, settings
from app.core.exceptions import ServiceConfigurationError
from app.db.session import get_db
from app.providers.embedding.openai_compatible import OpenAICompatibleEmbeddingProvider
from app.providers.keyword_search.base import KeywordSearchProvider
from app.providers.keyword_search.elasticsearch import ElasticsearchKeywordSearchProvider
from app.providers.llm.openai_compatible import OpenAICompatibleLLMProvider
from app.providers.query.llm_rewrite import LLMRewriteProcessor
from app.providers.query.pipeline import QueryPipeline
from app.providers.rerank.base import RerankProvider
from app.providers.rerank.llm import LLMRerankProvider
from app.providers.rerank.noop import NoopRerankProvider
from app.providers.vectorstores.pgvector import PgVectorStore
from app.repositories.document_repository import DocumentRepository
from app.repositories.knowledge_base_repository import KnowledgeBaseRepository
from app.repositories.retrieval_log_repository import RetrievalLogRepository
from app.repositories.tenant_repository import TenantRepository
from app.services.document_service import DocumentService
from app.services.feedback_service import FeedbackService
from app.services.indexing_service import IndexingService, build_indexing_service
from app.services.knowledge_base_service import KnowledgeBaseService
from app.services.quota_service import QuotaService
from app.services.rag_service import RagService
from app.services.rate_limit_service import RateLimitService
from app.tenant.plan_resolver import PlanResolver


def get_settings() -> Settings:
    return settings


def get_llm_provider(app_settings: Settings) -> OpenAICompatibleLLMProvider:
    if app_settings.llm_provider != "openai_compatible":
        raise ServiceConfigurationError(
            internal_message=f"unsupported LLM_PROVIDER: {app_settings.llm_provider}",
            context={"provider": app_settings.llm_provider},
        )
    return OpenAICompatibleLLMProvider(app_settings)


def get_plan_resolver(
    session: AsyncSession = Depends(get_db),
) -> PlanResolver:
    return PlanResolver(TenantRepository(session))


def get_quota_service(
    session: AsyncSession = Depends(get_db),
) -> QuotaService:
    return QuotaService(
        knowledge_base_repository=KnowledgeBaseRepository(session),
        document_repository=DocumentRepository(session),
    )


async def get_rate_limit_service(
    app_settings: Settings = Depends(get_settings),
) -> AsyncIterator[RateLimitService]:
    service = RateLimitService(redis_url=app_settings.celery_broker_url)
    try:
        yield service
    finally:
        await service.close()


def get_rerank_provider(app_settings: Settings) -> RerankProvider:
    if app_settings.rerank_provider == "noop":
        return NoopRerankProvider()
    if app_settings.rerank_provider == "llm":
        return LLMRerankProvider.from_settings(
            get_llm_provider(app_settings),
            app_settings,
        )
    raise ServiceConfigurationError(
        internal_message=f"unsupported RERANK_PROVIDER: {app_settings.rerank_provider}",
        context={"provider": app_settings.rerank_provider},
    )


def get_keyword_search_provider(app_settings: Settings) -> KeywordSearchProvider:
    if app_settings.keyword_search_provider == "elasticsearch":
        return ElasticsearchKeywordSearchProvider(app_settings)
    raise ServiceConfigurationError(
        internal_message=(
            "unsupported KEYWORD_SEARCH_PROVIDER: "
            f"{app_settings.keyword_search_provider}"
        ),
        context={"provider": app_settings.keyword_search_provider},
    )


def get_indexing_service(
    session: AsyncSession = Depends(get_db),
    app_settings: Settings = Depends(get_settings),
) -> IndexingService:
    return build_indexing_service(session, app_settings)


def get_knowledge_base_service(
    session: AsyncSession = Depends(get_db),
    indexing_service: IndexingService = Depends(get_indexing_service),
    plan_resolver: PlanResolver = Depends(get_plan_resolver),
    quota_service: QuotaService = Depends(get_quota_service),
) -> KnowledgeBaseService:
    return KnowledgeBaseService(
        session=session,
        repository=KnowledgeBaseRepository(session),
        document_repository=DocumentRepository(session),
        indexing_service=indexing_service,
        plan_resolver=plan_resolver,
        quota_service=quota_service,
    )


def get_document_service(
    session: AsyncSession = Depends(get_db),
    indexing_service: IndexingService = Depends(get_indexing_service),
    plan_resolver: PlanResolver = Depends(get_plan_resolver),
    quota_service: QuotaService = Depends(get_quota_service),
) -> DocumentService:
    return DocumentService(
        session=session,
        document_repository=DocumentRepository(session),
        knowledge_base_repository=KnowledgeBaseRepository(session),
        indexing_service=indexing_service,
        plan_resolver=plan_resolver,
        quota_service=quota_service,
    )


def get_rag_service(
    session: AsyncSession = Depends(get_db),
    app_settings: Settings = Depends(get_settings),
    plan_resolver: PlanResolver = Depends(get_plan_resolver),
    rate_limit_service: RateLimitService = Depends(get_rate_limit_service),
) -> RagService:
    return RagService(
        session=session,
        settings=app_settings,
        knowledge_base_repository=KnowledgeBaseRepository(session),
        retrieval_log_repository=RetrievalLogRepository(session),
        embedding_provider=OpenAICompatibleEmbeddingProvider(app_settings),
        vector_store=PgVectorStore(session),
        keyword_search_provider_factory=lambda: get_keyword_search_provider(app_settings),
        rerank_provider=get_rerank_provider(app_settings),
        plan_resolver=plan_resolver,
        rate_limit_service=rate_limit_service,
        query_pipeline=QueryPipeline(
            rewrite_enabled=app_settings.query_rewrite_enabled,
            rewrite_processor=LLMRewriteProcessor(
                get_llm_provider(app_settings),
                timeout_ms=app_settings.query_rewrite_timeout_ms,
            ),
        ),
    )


def get_feedback_service(
    session: AsyncSession = Depends(get_db),
    app_settings: Settings = Depends(get_settings),
) -> FeedbackService:
    return FeedbackService(
        settings=app_settings,
        retrieval_log_repository=RetrievalLogRepository(session),
    )
