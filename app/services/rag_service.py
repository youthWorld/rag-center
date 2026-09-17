import time
from collections.abc import Callable
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.error_codes import ErrorCode
from app.core.exceptions import (
    KnowledgeBaseNotFoundError,
    ServiceConfigurationError,
    raise_app_error,
)
from app.core.logging import get_logger
from app.observability.langfuse_client import RetrieveObservability
from app.providers.embedding.base import EmbeddingProvider
from app.providers.keyword_search.base import KeywordSearchProvider
from app.providers.query.pipeline import QueryPipeline
from app.providers.rerank.base import RerankProvider
from app.providers.rerank.noop import NoopRerankProvider
from app.providers.vectorstores.base import VectorStore
from app.repositories.knowledge_base_repository import KnowledgeBaseRepository
from app.repositories.retrieval_log_repository import RetrievalLogRepository
from app.schemas.hybrid_search import RetrievalMode, RetrievalOptions
from app.schemas.rag import QueryOptions, RagRetrieveRequest, RagRetrieveResponse, RetrievedChunk
from app.services.hybrid_search_service import HybridSearchService
from app.services.rate_limit_service import RateLimitService
from app.tenant.plan_resolver import PlanContext, PlanResolver
from app.tenant.retrieve_presets import expand_retrieve_profile
from app.utils.id_generator import generate_id


class RagService:
    def __init__(
        self,
        session: AsyncSession,
        settings: Settings,
        knowledge_base_repository: KnowledgeBaseRepository,
        retrieval_log_repository: RetrievalLogRepository,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        keyword_search_provider: KeywordSearchProvider | None = None,
        keyword_search_provider_factory: Callable[[], KeywordSearchProvider] | None = None,
        hybrid_search_service: HybridSearchService | None = None,
        rerank_provider: RerankProvider | None = None,
        query_pipeline: QueryPipeline | None = None,
        plan_resolver: PlanResolver | None = None,
        rate_limit_service: RateLimitService | None = None,
    ) -> None:
        self.session = session
        self.settings = settings
        self.knowledge_base_repository = knowledge_base_repository
        self.retrieval_log_repository = retrieval_log_repository
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.keyword_search_provider = keyword_search_provider
        self.keyword_search_provider_factory = keyword_search_provider_factory
        self.hybrid_search_service = hybrid_search_service or HybridSearchService(
            rrf_k=settings.hybrid_rrf_k
        )
        self.rerank_provider = rerank_provider or NoopRerankProvider()
        self.query_pipeline = query_pipeline or QueryPipeline(
            rewrite_enabled=settings.query_rewrite_enabled
        )
        self.plan_resolver = plan_resolver
        self.rate_limit_service = rate_limit_service
        self.logger = get_logger(__name__)

    async def retrieve(
        self,
        request: RagRetrieveRequest,
        *,
        tenant_id: str,
        tenant: Any | None = None,
        plan_context: PlanContext | None = None,
    ) -> RagRetrieveResponse:
        self.logger.info(
            "BUSINESS_EVENT | event=rag_retrieval_started | kb_id=%s | tenant_id=%s | user_id=%s",
            request.kb_id,
            tenant_id,
            request.user_id,
        )
        plan = plan_context or await self._resolve_plan_context(tenant_id, tenant=tenant)
        profile = request.profile
        if profile is None:
            profile = (
                "balanced"
                if plan_context is not None or tenant is not None or self._policy_is_configured()
                else "custom"
            )
        if profile not in plan.features.allowed_profiles:
            self._raise_feature_not_allowed(
                plan,
                profile=profile,
                feature="retrieve profile",
            )
        effective_request = self._expand_profile(request, profile)
        mode = self._resolve_retrieval_mode(effective_request)
        rerank_enabled = self._resolve_rerank_enabled(effective_request)
        query_rewrite_enabled = self._resolve_query_rewrite_enabled(effective_request)
        self._enforce_plan_features(
            plan,
            profile=profile,
            mode=mode,
            rerank_enabled=rerank_enabled,
            query_rewrite_enabled=query_rewrite_enabled,
        )

        observability = RetrieveObservability(
            settings=self.settings,
            tenant_id=tenant_id,
            kb_id=request.kb_id,
            user_id=request.user_id,
            profile=profile,
            plan=plan.plan,
            raw_query=request.query,
        )
        with observability:
            return await self._retrieve(
                request,
                tenant_id=tenant_id,
                plan=plan,
                profile=profile,
                effective_request=effective_request,
                mode=mode,
                rerank_enabled=rerank_enabled,
                query_rewrite_enabled=query_rewrite_enabled,
                observability=observability,
            )

    async def _retrieve(
        self,
        request: RagRetrieveRequest,
        *,
        tenant_id: str,
        plan: PlanContext,
        profile: str,
        effective_request: RagRetrieveRequest,
        mode: RetrievalMode,
        rerank_enabled: bool,
        query_rewrite_enabled: bool,
        observability: RetrieveObservability,
    ) -> RagRetrieveResponse:
        if self.rate_limit_service is not None:
            await self.rate_limit_service.check_retrieve(tenant_id, plan)

        knowledge_base = await self.knowledge_base_repository.get_by_id(
            kb_id=request.kb_id,
            tenant_id=tenant_id,
        )
        if knowledge_base is None:
            raise KnowledgeBaseNotFoundError()

        query_processing = await self.query_pipeline.process(
            request.query,
            knowledge_base=knowledge_base,
            query_options=effective_request.query_options,
        )
        search_query = query_processing.search_query
        observability.record_query_processing(
            effective_query=query_processing.effective_query,
            search_query=search_query,
            rewrite_latency_ms=query_processing.rewrite_latency_ms,
            synonym_applied=query_processing.synonym_applied,
            synonym_expansions=query_processing.synonym_expansions,
            degraded=query_processing.degraded,
            degraded_reason=query_processing.degraded_reason,
        )
        started_at = time.perf_counter()
        vector_top_k, bm25_top_k, top_k, rrf_k = self._resolve_retrieval_options(
            effective_request,
            mode,
        )
        if mode == "hybrid":
            self.logger.info(
                "HYBRID_SEARCH_START | tenant_id=%s | kb_id=%s | query=%s | "
                "vector_top_k=%s | bm25_top_k=%s",
                tenant_id,
                knowledge_base.id,
                search_query,
                vector_top_k,
                bm25_top_k,
            )

        vector_results: list[dict[str, Any]] = []
        bm25_results: list[dict[str, Any]] = []
        if mode in {"vector", "hybrid"}:
            vector_started_at = time.perf_counter()
            query_vector = await self.embedding_provider.embed_query(search_query)
            vector_results = await self.vector_store.similarity_search(
                query_vector,
                tenant_id=tenant_id,
                kb_id=knowledge_base.id,
                top_k=vector_top_k,
            )
            self.logger.info(
                "VECTOR_SEARCH_SUCCESS | tenant_id=%s | kb_id=%s | query=%s | "
                "vector_top_k=%s | vector_count=%s | cost_ms=%s",
                tenant_id,
                knowledge_base.id,
                search_query,
                vector_top_k,
                len(vector_results),
                int((time.perf_counter() - vector_started_at) * 1000),
            )

        retrieval_degraded = False
        degraded_reason: str | None = None
        if mode in {"bm25", "hybrid"}:
            keyword_search_provider = self._get_keyword_search_provider()
            if keyword_search_provider is None:
                raise ServiceConfigurationError(
                    internal_message="keyword search provider is not configured",
                    context={"mode": mode},
                )
            bm25_started_at = time.perf_counter()
            try:
                bm25_results = await keyword_search_provider.keyword_search(
                    query=search_query,
                    tenant_id=tenant_id,
                    kb_id=knowledge_base.id,
                    top_k=bm25_top_k,
                )
            except Exception as exception:
                if mode != "hybrid":
                    raise
                retrieval_degraded = True
                degraded_reason = "bm25 search failed"
                self.logger.exception(
                    "BM25_SEARCH_FAILED | tenant_id=%s | kb_id=%s | query=%s | "
                    "vector_top_k=%s | bm25_top_k=%s | vector_count=%s | "
                    "bm25_count=%s | fused_count=%s | cost_ms=%s | error=%s",
                    tenant_id,
                    knowledge_base.id,
                    search_query,
                    vector_top_k,
                    bm25_top_k,
                    len(vector_results),
                    0,
                    0,
                    int((time.perf_counter() - bm25_started_at) * 1000),
                    str(exception) or type(exception).__name__,
                )
                self.logger.warning(
                    "HYBRID_SEARCH_DEGRADED | tenant_id=%s | kb_id=%s | query=%s | "
                    "degraded_reason=%s | vector_count=%s | cost_ms=%s",
                    tenant_id,
                    knowledge_base.id,
                    search_query,
                    degraded_reason,
                    len(vector_results),
                    int((time.perf_counter() - started_at) * 1000),
                )
            else:
                self.logger.info(
                    "BM25_SEARCH_SUCCESS | tenant_id=%s | kb_id=%s | query=%s | "
                    "bm25_top_k=%s | bm25_count=%s | cost_ms=%s",
                    tenant_id,
                    knowledge_base.id,
                    search_query,
                    bm25_top_k,
                    len(bm25_results),
                    int((time.perf_counter() - bm25_started_at) * 1000),
                )

        if mode == "vector":
            retrieved = self.hybrid_search_service.normalize_vector_results(vector_results)
            fused_count = len(retrieved)
        elif mode == "bm25":
            retrieved = self.hybrid_search_service.normalize_bm25_results(bm25_results)
            fused_count = len(retrieved)
        elif retrieval_degraded:
            retrieved = self.hybrid_search_service.normalize_vector_results(vector_results)
            fused_count = len(retrieved)
        else:
            fused_started_at = time.perf_counter()
            retrieved = self.hybrid_search_service.fuse(
                vector_results,
                bm25_results,
                rrf_k=rrf_k,
            )
            fused_count = len(retrieved)
            self.logger.info(
                "RRF_FUSION_SUCCESS | tenant_id=%s | kb_id=%s | query=%s | "
                "vector_count=%s | bm25_count=%s | fused_count=%s | cost_ms=%s",
                tenant_id,
                knowledge_base.id,
                search_query,
                len(vector_results),
                len(bm25_results),
                fused_count,
                int((time.perf_counter() - fused_started_at) * 1000),
            )

        retrieved = retrieved[:top_k]
        observability.record_retrieval(
            search_query=search_query,
            mode=mode,
            vector_count=len(vector_results),
            bm25_count=len(bm25_results),
            fused_count=fused_count,
            degraded=retrieval_degraded,
            degraded_reason=degraded_reason,
        )

        rerank_top_n = self._resolve_rerank_top_n(effective_request)
        candidate_count = 0
        rerank_degraded = False
        rerank_error: str | None = None
        reranked = retrieved
        if rerank_enabled:
            rerank_candidates = retrieved[: self.settings.rerank_max_candidates]
            candidate_count = len(rerank_candidates)
            try:
                reranked = await self.rerank_provider.rerank(
                    query=effective_request.query,
                    chunks=rerank_candidates,
                    top_n=rerank_top_n,
                )
            except Exception as exception:
                rerank_degraded = True
                rerank_error = str(exception) or type(exception).__name__
                reranked = retrieved
                self.logger.exception(
                    "BUSINESS_EVENT | event=rag_rerank_degraded | kb_id=%s | "
                    "candidate_count=%s | error=%s",
                    knowledge_base.id,
                    candidate_count,
                    rerank_error,
                )

        observability.record_rerank(
            enabled=rerank_enabled,
            candidate_count=candidate_count,
            degraded=rerank_degraded,
            error=rerank_error,
        )

        latency_ms = int((time.perf_counter() - started_at) * 1000)
        response_chunks = (
            reranked[:rerank_top_n]
            if rerank_enabled and not rerank_degraded
            else reranked
        )
        retrieved_chunks = [
            RetrievedChunk(
                document_id=item["document_id"],
                chunk_id=item["chunk_id"],
                title=item["title"],
                content=item["content"],
                score=float(item["score"]),
                vector_score=self._optional_float(item.get("vector_score")),
                bm25_score=self._optional_float(item.get("bm25_score")),
                vector_rank=item.get("vector_rank"),
                bm25_rank=item.get("bm25_rank"),
                retrieval_source=item.get("retrieval_source", "vector"),
                rerank_score=(
                    float(item["rerank_score"])
                    if item.get("rerank_score") is not None
                    else None
                ),
            )
            for item in response_chunks
        ]
        serialized_chunks = [chunk.model_dump() for chunk in retrieved_chunks]
        retrieval_log = await self.retrieval_log_repository.create(
            tenant_id=tenant_id,
            kb_id=knowledge_base.id,
            user_id=request.user_id,
            query=request.query,
            trace_id=observability.trace_id,
            profile=profile,
            search_query=search_query,
            effective_query=query_processing.effective_query,
            retrieved_chunks=serialized_chunks,
            top_k=top_k,
            vector_store=self.settings.vector_store,
            latency_ms=latency_ms,
        )
        raw_log_id = getattr(retrieval_log, "id", None)
        log_id = raw_log_id if isinstance(raw_log_id, str) and raw_log_id else generate_id()
        await self.session.commit()
        observability.finish(log_id=log_id, chunks=serialized_chunks)
        if self.rate_limit_service is not None:
            await self.rate_limit_service.record_retrieve_success(tenant_id)
        self.logger.info(
            "BUSINESS_EVENT | event=rag_retrieval_completed | tenant_id=%s | "
            "kb_id=%s | query=%s | vector_top_k=%s | bm25_top_k=%s | "
            "vector_count=%s | bm25_count=%s | fused_count=%s | chunk_count=%s | "
            "latency_ms=%s | cost_ms=%s",
            tenant_id,
            knowledge_base.id,
            search_query,
            vector_top_k,
            bm25_top_k,
            len(vector_results),
            len(bm25_results),
            fused_count,
            len(retrieved_chunks),
            latency_ms,
            latency_ms,
        )

        return RagRetrieveResponse(
            query=request.query,
            kb_id=knowledge_base.id,
            retrieved_chunks=retrieved_chunks,
            metadata={
                "log_id": log_id,
                "trace_id": observability.trace_id,
                "top_k": top_k,
                "latency_ms": latency_ms,
                "vector_store": self.settings.vector_store,
                "query_processing": (
                    query_processing.to_dict()
                    if query_processing.should_expose()
                    else None
                ),
                "retrieval": self._build_retrieval_metadata(
                    mode=mode,
                    rrf_k=rrf_k,
                    vector_top_k=vector_top_k,
                    bm25_top_k=bm25_top_k,
                    vector_count=len(vector_results),
                    bm25_count=len(bm25_results),
                    fused_count=fused_count,
                    degraded=retrieval_degraded,
                    degraded_reason=degraded_reason,
                ),
                "rerank": self._build_rerank_metadata(
                    enabled=rerank_enabled,
                    top_n=rerank_top_n,
                    candidate_count=candidate_count,
                    degraded=rerank_degraded,
                    error=rerank_error,
                ),
                "tenant_policy": {
                    "plan": plan.plan,
                    "retrieve_profile": profile,
                    "effective_mode": mode,
                    "effective_rerank": rerank_enabled,
                    "effective_query_rewrite": query_rewrite_enabled,
                },
            },
        )

    async def _resolve_plan_context(
        self,
        tenant_id: str,
        *,
        tenant: Any | None = None,
    ) -> PlanContext:
        if tenant is not None and self.plan_resolver is not None:
            resolve = getattr(self.plan_resolver, "resolve", None)
            if resolve is not None:
                return resolve(tenant)
        if tenant is not None:
            from app.tenant.plan_resolver import resolve_plan

            return resolve_plan(tenant)
        if self.plan_resolver is not None:
            resolve_for_tenant_id = getattr(
                self.plan_resolver,
                "resolve_for_tenant_id",
                None,
            )
            if resolve_for_tenant_id is not None:
                return await resolve_for_tenant_id(tenant_id)
        return await PlanResolver().resolve_for_tenant_id(tenant_id)

    def _policy_is_configured(self) -> bool:
        return self.plan_resolver is not None or self.rate_limit_service is not None

    @staticmethod
    def _expand_profile(
        request: RagRetrieveRequest,
        profile: str,
    ) -> RagRetrieveRequest:
        if profile == "custom":
            return request
        preset = expand_retrieve_profile(profile)
        updates: dict[str, Any] = {}
        if "top_k" in preset:
            updates["top_k"] = preset["top_k"]
        if "retrieval_options" in preset:
            updates["retrieval_options"] = RetrievalOptions.model_validate(
                preset["retrieval_options"]
            )
        if "rerank_options" in preset:
            from app.schemas.rerank import RerankOptions

            updates["rerank_options"] = RerankOptions.model_validate(preset["rerank_options"])
        if "query_options" in preset:
            updates["query_options"] = QueryOptions.model_validate(preset["query_options"])
        return request.model_copy(update=updates)

    def _enforce_plan_features(
        self,
        plan: PlanContext,
        *,
        profile: str,
        mode: RetrievalMode,
        rerank_enabled: bool,
        query_rewrite_enabled: bool,
    ) -> None:
        if mode == "hybrid" and not plan.features.hybrid_allowed:
            self._raise_feature_not_allowed(plan, profile=profile, feature="hybrid retrieval")
        if rerank_enabled and not plan.features.rerank_allowed:
            self._raise_feature_not_allowed(plan, profile=profile, feature="rerank")
        if query_rewrite_enabled and not plan.features.query_rewrite_allowed:
            self._raise_feature_not_allowed(plan, profile=profile, feature="query rewrite")

    @staticmethod
    def _raise_feature_not_allowed(plan: PlanContext, *, profile: str, feature: str) -> None:
        raise_app_error(
            ErrorCode.FEATURE_NOT_ALLOWED,
            f"{feature} is not allowed for the {plan.plan} plan",
            data={"plan": plan.plan, "profile": profile, "feature": feature},
        )

    def _resolve_retrieval_mode(self, request: RagRetrieveRequest) -> RetrievalMode:
        requested_mode = (
            request.retrieval_options.mode
            if request.retrieval_options is not None
            else None
        )
        mode = requested_mode or self.settings.retrieval_mode
        if mode not in {"vector", "bm25", "hybrid"}:
            raise ServiceConfigurationError(
                internal_message=f"unsupported RETRIEVAL_MODE: {mode}",
                context={"mode": mode},
            )
        if mode == "hybrid" and self.settings.hybrid_fusion != "rrf":
            raise ServiceConfigurationError(
                internal_message=f"unsupported HYBRID_FUSION: {self.settings.hybrid_fusion}",
                context={"fusion": self.settings.hybrid_fusion},
            )
        return mode

    def _get_keyword_search_provider(self) -> KeywordSearchProvider | None:
        if (
            self.keyword_search_provider is None
            and self.keyword_search_provider_factory is not None
        ):
            self.keyword_search_provider = self.keyword_search_provider_factory()
        return self.keyword_search_provider

    def _resolve_retrieval_options(
        self,
        request: RagRetrieveRequest,
        mode: RetrievalMode,
    ) -> tuple[int, int, int, int]:
        options = request.retrieval_options
        requested_top_k = request.top_k if request.top_k is not None else self.settings.top_k

        if mode == "hybrid":
            vector_top_k = (
                options.vector_top_k
                if options is not None and options.vector_top_k is not None
                else (
                    request.top_k
                    if request.top_k is not None
                    else self.settings.hybrid_vector_top_k
                )
            )
            bm25_top_k = (
                options.bm25_top_k
                if options is not None and options.bm25_top_k is not None
                else (
                    request.top_k
                    if request.top_k is not None
                    else self.settings.hybrid_bm25_top_k
                )
            )
            top_k = request.top_k if request.top_k is not None else self.settings.hybrid_top_n
        elif mode == "bm25":
            vector_top_k = 0
            bm25_top_k = (
                options.bm25_top_k
                if options is not None and options.bm25_top_k is not None
                else requested_top_k
            )
            top_k = requested_top_k
        else:
            vector_top_k = (
                options.vector_top_k
                if options is not None and options.vector_top_k is not None
                else requested_top_k
            )
            bm25_top_k = 0
            top_k = requested_top_k

        rrf_k = (
            options.rrf_k
            if options is not None and options.rrf_k is not None
            else self.settings.hybrid_rrf_k
        )
        return vector_top_k, bm25_top_k, top_k, rrf_k

    def _build_retrieval_metadata(
        self,
        *,
        mode: RetrievalMode,
        rrf_k: int,
        vector_top_k: int,
        bm25_top_k: int,
        vector_count: int,
        bm25_count: int,
        fused_count: int,
        degraded: bool,
        degraded_reason: str | None,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "mode": mode,
            "fusion": "rrf" if mode == "hybrid" else "none",
            "rrf_k": rrf_k if mode == "hybrid" else None,
            "vector_store": self.settings.vector_store,
            "keyword_search": (
                self.settings.keyword_search_provider
                if mode in {"bm25", "hybrid"}
                else None
            ),
            "vector_top_k": vector_top_k,
            "bm25_top_k": bm25_top_k,
            "vector_count": vector_count,
            "bm25_count": bm25_count,
            "fused_count": fused_count,
        }
        if degraded:
            metadata["degraded"] = True
            metadata["degraded_reason"] = degraded_reason or "bm25 search failed"
        return metadata

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        return float(value) if value is not None else None

    def _resolve_rerank_enabled(self, request: RagRetrieveRequest) -> bool:
        if request.rerank_options is not None and request.rerank_options.enabled is not None:
            return request.rerank_options.enabled
        return self.settings.rerank_enabled

    def _resolve_query_rewrite_enabled(self, request: RagRetrieveRequest) -> bool:
        options = request.query_options
        if options is None:
            return self.settings.query_rewrite_enabled
        if options.strategy == "noop":
            return False
        if options.enabled is not None:
            return options.enabled
        if options.strategy == "rewrite":
            return True
        return self.settings.query_rewrite_enabled

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
