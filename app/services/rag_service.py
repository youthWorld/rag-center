import asyncio
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.error_codes import ErrorCode
from app.core.exceptions import (
    AppError,
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
from app.repositories.index_version_repository import IndexVersionRepository
from app.repositories.knowledge_base_repository import KnowledgeBaseRepository
from app.schemas.hybrid_search import RetrievalMode, RetrievalOptions
from app.schemas.rag import (
    MULTI_KB_MAX,
    EvidenceOptions,
    QueryOptions,
    RagRetrieveRequest,
    RagRetrieveResponse,
    RetrievedChunk,
    RetrieveEvidenceMetadata,
)
from app.services.context_expansion_service import ContextExpansionService
from app.services.evidence_orchestration_service import (
    EvidenceOrchestrationResult,
    EvidenceOrchestrationService,
)
from app.services.graph_candidate_expansion_service import GraphCandidateExpansionService
from app.services.hybrid_search_service import HybridSearchService
from app.services.rate_limit_service import RateLimitService
from app.services.retrieval_usage import (
    record_request,
    record_retrieval_model_call,
    retrieval_requests,
)
from app.tenant.plan_resolver import PlanContext, PlanResolver
from app.tenant.retrieve_presets import expand_retrieve_profile
from app.utils.id_generator import generate_id
from app.utils.markdown_splitter import is_low_information_content, normalize_content


@dataclass(slots=True)
class _RetrievalCandidates:
    chunks: list[dict[str, Any]]
    vector_count: int
    bm25_count: int
    fused_count: int
    degraded: bool = False
    degraded_reason: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


class RagService:
    def __init__(
        self,
        session: AsyncSession,
        settings: Settings,
        knowledge_base_repository: KnowledgeBaseRepository,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        keyword_search_provider: KeywordSearchProvider | None = None,
        keyword_search_provider_factory: Callable[[], KeywordSearchProvider] | None = None,
        hybrid_search_service: HybridSearchService | None = None,
        rerank_provider: RerankProvider | None = None,
        query_pipeline: QueryPipeline | None = None,
        plan_resolver: PlanResolver | None = None,
        rate_limit_service: RateLimitService | None = None,
        graph_candidate_expansion_service: GraphCandidateExpansionService | None = None,
        context_expansion_service: ContextExpansionService | None = None,
        evidence_orchestration_service: EvidenceOrchestrationService | None = None,
        index_version_repository: IndexVersionRepository | None = None,
    ) -> None:
        self.session = session
        self.settings = settings
        self.knowledge_base_repository = knowledge_base_repository
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
        self.graph_candidate_expansion_service = (
            graph_candidate_expansion_service or GraphCandidateExpansionService(session)
        )
        self.context_expansion_service = context_expansion_service or ContextExpansionService(
            session
        )
        self.evidence_orchestration_service = evidence_orchestration_service
        self.index_version_repository = index_version_repository or IndexVersionRepository(session)
        self.logger = get_logger(__name__)

    async def retrieve(
        self,
        request: RagRetrieveRequest,
        *,
        tenant_id: str,
        tenant: Any | None = None,
        plan_context: PlanContext | None = None,
    ) -> RagRetrieveResponse:
        log_id = generate_id()
        self._validate_query(request.query)
        kb_ids = self._resolve_kb_ids(request)
        self.logger.info(
            "BUSINESS_EVENT | event=rag_retrieval_started | kb_ids=%s | tenant_id=%s | user_id=%s",
            kb_ids,
            tenant_id,
            request.user_id,
        )
        plan = plan_context or await self._resolve_plan_context(tenant_id, tenant=tenant)
        profile = request.profile or "balanced"
        self._enforce_multi_kb_limit(plan, profile=profile, kb_ids=kb_ids)
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
        evidence_options = self._resolve_evidence_options(effective_request)
        self._enforce_plan_features(
            plan,
            profile=profile,
            mode=mode,
            rerank_enabled=rerank_enabled,
            query_rewrite_enabled=query_rewrite_enabled,
            evidence_enabled=bool(evidence_options.enabled),
        )

        observability = RetrieveObservability(
            settings=self.settings,
            tenant_id=tenant_id,
            kb_id=kb_ids[0],
            kb_ids=kb_ids if len(kb_ids) > 1 else None,
            user_id=request.user_id,
            profile=profile,
            plan=plan.plan,
            raw_query=request.query,
            log_id=log_id,
            enabled=request.observability_enabled is not False,
        )
        with observability:
            return await self._retrieve(
                request,
                kb_ids=kb_ids,
                tenant_id=tenant_id,
                plan=plan,
                profile=profile,
                effective_request=effective_request,
                mode=mode,
                rerank_enabled=rerank_enabled,
                query_rewrite_enabled=query_rewrite_enabled,
                evidence_options=evidence_options,
                observability=observability,
                log_id=log_id,
            )

    async def _retrieve(
        self,
        request: RagRetrieveRequest,
        *,
        kb_ids: list[str],
        tenant_id: str,
        plan: PlanContext,
        profile: str,
        effective_request: RagRetrieveRequest,
        mode: RetrievalMode,
        rerank_enabled: bool,
        query_rewrite_enabled: bool,
        evidence_options: EvidenceOptions,
        observability: RetrieveObservability,
        internal_retrieve_once: bool = False,
        frozen_index_versions: dict[str, str] | None = None,
        log_id: str | None = None,
    ) -> RagRetrieveResponse:
        if self.rate_limit_service is not None and not internal_retrieve_once:
            await self.rate_limit_service.check_retrieve(tenant_id, plan)

        return await self._retrieve_scope(
            request,
            kb_ids=list(dict.fromkeys(kb_ids)),
            tenant_id=tenant_id,
            plan=plan,
            profile=profile,
            effective_request=effective_request,
            mode=mode,
            rerank_enabled=rerank_enabled,
            query_rewrite_enabled=query_rewrite_enabled,
            evidence_options=evidence_options,
            observability=observability,
            internal_retrieve_once=internal_retrieve_once,
            frozen_index_versions=frozen_index_versions,
            log_id=log_id,
        )

    async def _retrieve_scope(
        self,
        request: RagRetrieveRequest,
        *,
        kb_ids: list[str],
        tenant_id: str,
        plan: PlanContext,
        profile: str,
        effective_request: RagRetrieveRequest,
        mode: RetrievalMode,
        rerank_enabled: bool,
        query_rewrite_enabled: bool,
        evidence_options: EvidenceOptions,
        observability: RetrieveObservability,
        internal_retrieve_once: bool = False,
        frozen_index_versions: dict[str, str] | None = None,
        log_id: str | None = None,
    ) -> RagRetrieveResponse:
        knowledge_bases = await self._load_knowledge_bases(
            kb_ids=kb_ids,
            tenant_id=tenant_id,
        )
        if frozen_index_versions is not None:
            versions = {
                str(kb.id): frozen_index_versions.get(str(kb.id), "") for kb in knowledge_bases
            }
            missing = [kb_id for kb_id, version in versions.items() if not version]
            if missing:
                raise ServiceConfigurationError(
                    internal_message="frozen index version is missing",
                    context={"kb_ids": missing},
                )
        else:
            versions = {
                str(kb.id): await self._resolve_index_version(
                    request.index_version, knowledge_base=kb, tenant_id=tenant_id
                )
                for kb in knowledge_bases
            }
        query_processing = await self.query_pipeline.process(
            request.query,
            knowledge_base=min(knowledge_bases, key=lambda kb: str(kb.id)),
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
        keyword_search_provider = (
            self._get_keyword_search_provider() if mode in {"bm25", "hybrid"} else None
        )
        candidate = await self._retrieve_candidates(
            tenant_id=tenant_id,
            index_versions=versions,
            search_query=search_query,
            mode=mode,
            vector_top_k=vector_top_k,
            bm25_top_k=bm25_top_k,
            rrf_k=rrf_k,
            keyword_search_provider=keyword_search_provider,
        )
        names = {str(kb.id): getattr(kb, "name", None) for kb in knowledge_bases}
        raw = [{**item, "kb_name": names[item["kb_id"]]} for item in candidate.chunks]
        direct_retrieved, filter_metadata = self._filter_scope_candidates(raw)
        base_candidate_snapshot = [dict(item) for item in direct_retrieved]
        if len(versions) == 1:
            base_candidate_snapshot = base_candidate_snapshot[: max(top_k, 20)]
        vector_count, bm25_count = candidate.vector_count, candidate.bm25_count
        fused_count = candidate.fused_count
        degraded, degraded_reason = candidate.degraded, candidate.degraded_reason
        base_input = direct_retrieved[: 20 if profile == "quality" else max(top_k, 20)]
        graph_result = await self.graph_candidate_expansion_service.expand_candidates(
            tenant_id=tenant_id,
            kb_ids=kb_ids,
            seeds=base_input,
            max_injected=None if internal_retrieve_once else 10,
        )
        self._validate_scope(graph_result.candidates, tenant_id, versions)
        if not internal_retrieve_once:
            graph_result.candidates = self._filter_scope_candidates(
                self._merge_graph_candidates(base_candidate_snapshot, graph_result.candidates)
            )[0][len(base_candidate_snapshot) :][:10]
            graph_result.injected_count = len(graph_result.candidates)
        retrieved = self._merge_graph_candidates(base_input, graph_result.candidates)
        if profile != "quality":
            retrieved = retrieved[: max(top_k, 30)]
        empty_reason = (
            await self._resolve_empty_reason(
                kb_ids=kb_ids,
                tenant_id=tenant_id,
            )
            if not retrieved
            else None
        )
        retrieval_metadata = self._build_retrieval_metadata(
            mode=mode,
            rrf_k=rrf_k,
            vector_top_k=vector_top_k,
            bm25_top_k=bm25_top_k,
            vector_count=vector_count,
            bm25_count=bm25_count,
            fused_count=fused_count,
            degraded=degraded,
            degraded_reason=degraded_reason,
            empty_reason=empty_reason,
            multi_kb=len(versions) > 1,
            kb_count=len(kb_ids),
        )
        retrieval_metadata.update(candidate.metrics)
        retrieval_metadata.update(filter_metadata)
        retrieval_metadata.update(
            {
                "kb_ids": sorted(versions),
                "kb_count": len(versions),
                "fusion_scope": "global",
                "base_candidate_count": len(base_input),
                "rerank_input_count": len(retrieved) if rerank_enabled else 0,
                "index_versions": versions,
                "graph_injected_count": graph_result.injected_count,
                "graph_injection_latency_ms": graph_result.latency_ms,
            }
        )
        observability.record_retrieval(
            search_query=search_query,
            mode=mode,
            vector_count=vector_count,
            bm25_count=bm25_count,
            fused_count=fused_count,
            degraded=degraded,
            degraded_reason=degraded_reason,
            empty_reason=empty_reason,
        )

        rerank_top_n = min(self._resolve_rerank_top_n(effective_request), top_k, len(retrieved))
        candidate_count = 0
        rerank_degraded = False
        rerank_error: str | None = None
        reranked = retrieved
        rerank_started = time.perf_counter()
        if rerank_enabled and retrieved:
            rerank_candidates = retrieved[:50]
            candidate_count = len(rerank_candidates)
            try:
                if not isinstance(self.rerank_provider, NoopRerankProvider):
                    record_retrieval_model_call("rerank")
                reranked = await self.rerank_provider.rerank(
                    query=search_query,
                    chunks=rerank_candidates,
                    top_n=rerank_top_n,
                )
            except Exception as exception:
                rerank_degraded = True
                rerank_error = self._safe_rerank_error(exception)
                reranked = [
                    {**item, "rerank_score": None}
                    for item in (base_input if profile == "quality" else rerank_candidates)
                ]
                self.logger.warning(
                    "BUSINESS_EVENT | event=rag_rerank_degraded | kb_ids=%s | "
                    "candidate_count=%s | error=%s",
                    kb_ids,
                    candidate_count,
                    rerank_error,
                )

        rerank_latency_ms = (
            int((time.perf_counter() - rerank_started) * 1000) if candidate_count else 0
        )
        rerank_model_calls = int(
            rerank_enabled
            and candidate_count > 0
            and not isinstance(self.rerank_provider, NoopRerankProvider)
        )
        application_model_call_details = {
            "embedding": candidate.metrics["embedding_request_count"],
            "query_rewrite": query_processing.application_model_calls,
            "rerank": rerank_model_calls,
        }
        observability.record_rerank(
            enabled=rerank_enabled,
            candidate_count=candidate_count,
            degraded=rerank_degraded,
            error=rerank_error,
        )

        graph_selected_count = 0
        if rerank_enabled:
            response_chunks = reranked[:rerank_top_n]
            graph_selected_count = sum(
                item.get("retrieval_source") == "graph" for item in response_chunks
            )
        else:
            response_chunks, graph_selected_count = self._select_without_rerank(
                seeds=direct_retrieved,
                injected=graph_result.candidates,
                top_k=top_k,
            )
        retrieval_metadata["graph_selected_count"] = graph_selected_count
        context_result = await self.context_expansion_service.expand(
            tenant_id=tenant_id, kb_ids=kb_ids, anchors=response_chunks
        )
        response_chunks = context_result.anchors
        self._validate_scope(response_chunks, tenant_id, versions)
        retrieval_metadata["source_distribution"] = {
            kb: sum(item["kb_id"] == kb for item in response_chunks) for kb in sorted(versions)
        }
        latency_ms = int((time.perf_counter() - started_at) * 1000)
        retrieved_chunks = [
            RetrievedChunk(
                document_id=item["document_id"],
                chunk_id=item["chunk_id"],
                kb_id=(str(item["kb_id"]) if item.get("kb_id") is not None else None),
                kb_name=(str(item["kb_name"]) if item.get("kb_name") is not None else None),
                title=item["title"],
                content=item["content"],
                score=float(item["score"]),
                vector_score=self._optional_float(item.get("vector_score")),
                bm25_score=self._optional_float(item.get("bm25_score")),
                vector_rank=item.get("vector_rank"),
                bm25_rank=item.get("bm25_rank"),
                retrieval_source=item.get("retrieval_source", "vector"),
                rerank_score=(
                    float(item["rerank_score"]) if item.get("rerank_score") is not None else None
                ),
                index_version=item.get("index_version") or versions[str(item["kb_id"])],
                section_id=item.get("section_id"),
                parent_section_id=item.get("parent_section_id"),
                order_index=item.get("order_index"),
                context=item.get("context"),
                metadata=dict(item.get("metadata") or {}),
            )
            for item in response_chunks
        ]
        evidence_result = await self._orchestrate_evidence(
            query=request.query,
            tenant_id=tenant_id,
            kb_ids=kb_ids,
            index_versions=versions,
            candidates=base_candidate_snapshot,
            retrieved_chunks=response_chunks,
            options=evidence_options,
        )
        evidence_metadata = evidence_result.metadata.model_dump(mode="json")
        if evidence_result.metadata.enabled:
            application_model_call_details["evidence"] = int(evidence_result.metadata.executed)
        serialized_chunks = [chunk.model_dump() for chunk in retrieved_chunks]
        resolved_log_id = log_id or generate_id()
        if not internal_retrieve_once:
            observability.finish(
                log_id=resolved_log_id,
                chunks=serialized_chunks,
                model_calls=application_model_call_details,
            )
            if self.rate_limit_service is not None:
                await self.rate_limit_service.record_retrieve_success(tenant_id)
        self.logger.info(
            "BUSINESS_EVENT | event=rag_retrieval_completed | tenant_id=%s | "
            "kb_ids=%s | query=%s | vector_top_k=%s | bm25_top_k=%s | "
            "vector_count=%s | bm25_count=%s | fused_count=%s | chunk_count=%s | "
            "latency_ms=%s | cost_ms=%s",
            tenant_id,
            kb_ids,
            search_query,
            vector_top_k,
            bm25_top_k,
            vector_count,
            bm25_count,
            fused_count,
            len(retrieved_chunks),
            latency_ms,
            latency_ms,
        )

        response = RagRetrieveResponse(
            query=request.query,
            kb_id=kb_ids[0],
            kb_ids=kb_ids,
            retrieved_chunks=retrieved_chunks,
            evidence_pack=evidence_result.pack,
            metadata={
                "log_id": resolved_log_id,
                "trace_id": observability.trace_id,
                "top_k": top_k,
                "latency_ms": latency_ms,
                "vector_store": self.settings.vector_store,
                "query_processing": (
                    query_processing.to_dict() if query_processing.should_expose() else None
                ),
                "retrieval": retrieval_metadata,
                "index_versions": versions,
                **({"index_version": next(iter(versions.values()))} if len(versions) == 1 else {}),
                "graph_injection": {
                    "executed": any(v != "v1" for v in versions.values()),
                    "graph_injected_count": graph_result.injected_count,
                    "graph_selected_count": graph_selected_count,
                    "latency_ms": graph_result.latency_ms,
                    "degraded": graph_result.degraded,
                    "error": graph_result.error,
                    "sources": graph_result.sources,
                },
                "context_expansion": {
                    "executed": any(v != "v1" for v in versions.values()),
                    "anchor_count": len(response_chunks),
                    "expanded_anchor_count": context_result.expanded_anchor_count,
                    "supplemental_chunk_count": context_result.supplemental_chunk_count,
                    "deduplicated_count": context_result.deduplicated_count,
                    "latency_ms": context_result.latency_ms,
                    "degraded": context_result.degraded,
                    "error": context_result.error,
                },
                "evidence": evidence_metadata,
                "rerank": self._build_rerank_metadata(
                    enabled=rerank_enabled,
                    top_n=rerank_top_n,
                    candidate_count=candidate_count,
                    returned_count=len(response_chunks),
                    latency_ms=rerank_latency_ms,
                    degraded=rerank_degraded,
                    error=rerank_error,
                ),
                "tenant_policy": {
                    "plan": plan.plan,
                    "retrieve_profile": profile,
                    "effective_mode": mode,
                    "effective_rerank": rerank_enabled,
                    "effective_query_rewrite": query_rewrite_enabled,
                    "effective_evidence": bool(evidence_options.enabled),
                },
                "application_model_calls": sum(application_model_call_details.values()),
                "application_model_call_details": application_model_call_details,
            },
        )
        if internal_retrieve_once:
            response._candidate_snapshot = base_candidate_snapshot
        return response

    async def _retrieve_candidates(
        self,
        *,
        tenant_id: str,
        index_versions: dict[str, str],
        search_query: str,
        mode: RetrievalMode,
        vector_top_k: int,
        bm25_top_k: int,
        rrf_k: int,
        keyword_search_provider: KeywordSearchProvider | None,
    ) -> _RetrievalCandidates:
        if mode in {"bm25", "hybrid"} and keyword_search_provider is None:
            raise ServiceConfigurationError(internal_message="keyword provider is not configured")
        metrics = {
            key: 0
            for key in (
                "embedding_request_count",
                "vector_request_count",
                "bm25_request_count",
                "embedding_latency_ms",
                "vector_store_latency_ms",
                "vector_latency_ms",
                "bm25_latency_ms",
                "hybrid_latency_ms",
            )
        }
        started = time.perf_counter()

        async def vector():
            branch_started = time.perf_counter()
            try:
                if not getattr(self.embedding_provider, "records_request_attempts", False):
                    record_request("embedding")
                embedding_started = time.perf_counter()
                try:
                    query_vector = await self.embedding_provider.embed_query(search_query)
                finally:
                    metrics["embedding_latency_ms"] = int(
                        (time.perf_counter() - embedding_started) * 1000
                    )
                sql_started = time.perf_counter()
                if not getattr(self.vector_store, "records_request_attempts", False):
                    record_request("vector")
                try:
                    return await self.vector_store.similarity_search_scope(
                        query_vector,
                        tenant_id=tenant_id,
                        index_versions=index_versions,
                        top_k=vector_top_k,
                    )
                finally:
                    metrics["vector_store_latency_ms"] = int(
                        (time.perf_counter() - sql_started) * 1000
                    )
            except Exception as exc:
                return exc
            finally:
                metrics["vector_latency_ms"] = int((time.perf_counter() - branch_started) * 1000)

        async def bm25():
            branch_started = time.perf_counter()
            try:
                if not getattr(keyword_search_provider, "records_request_attempts", False):
                    record_request("bm25")
                return await keyword_search_provider.keyword_search_scope(
                    query=search_query,
                    tenant_id=tenant_id,
                    index_versions=index_versions,
                    top_k=bm25_top_k,
                )
            except Exception as exc:
                return exc
            finally:
                metrics["bm25_latency_ms"] = int((time.perf_counter() - branch_started) * 1000)

        token = retrieval_requests.set(metrics)
        try:
            vector_result, bm25_result = [], []
            if mode == "hybrid":
                vector_result, bm25_result = await asyncio.gather(vector(), bm25())
            elif mode == "vector":
                vector_result = await vector()
            else:
                bm25_result = await bm25()
        finally:
            retrieval_requests.reset(token)
        vector_failed = isinstance(vector_result, Exception)
        bm25_failed = isinstance(bm25_result, Exception)
        for result in (vector_result, bm25_result):
            if isinstance(result, ServiceConfigurationError):
                raise result
            if not isinstance(result, Exception):
                self._validate_scope(result, tenant_id, index_versions)
        if (
            (mode == "vector" and vector_failed)
            or (mode == "bm25" and bm25_failed)
            or (vector_failed and bm25_failed)
        ):
            raise AppError(code=ErrorCode.RETRIEVAL_FAILED, context={"retrieval": metrics})
        vector_chunks = [] if vector_failed else vector_result
        bm25_chunks = [] if bm25_failed else bm25_result
        if mode == "hybrid" and not vector_failed and not bm25_failed:
            chunks = self.hybrid_search_service.fuse(vector_chunks, bm25_chunks, rrf_k=rrf_k)
            metrics["fusion"] = "rrf"
        elif mode == "bm25" or vector_failed:
            chunks = self.hybrid_search_service.normalize_bm25_results(bm25_chunks)
            metrics["fusion"] = "none"
        else:
            chunks = self.hybrid_search_service.normalize_vector_results(vector_chunks)
            metrics["fusion"] = "none"
        if mode == "hybrid":
            metrics["hybrid_latency_ms"] = int((time.perf_counter() - started) * 1000)
        return _RetrievalCandidates(
            chunks=chunks,
            vector_count=len(vector_chunks),
            bm25_count=len(bm25_chunks),
            fused_count=len(chunks),
            degraded=vector_failed or bm25_failed,
            degraded_reason="vector search failed"
            if vector_failed
            else "bm25 search failed"
            if bm25_failed
            else None,
            metrics=metrics,
        )

    @staticmethod
    def _validate_scope(chunks, tenant_id, versions):
        for item in chunks:
            if (
                not item.get("chunk_id")
                or not item.get("document_id")
                or item.get("kb_id") not in versions
                or item.get("index_version") != versions.get(item.get("kb_id"))
                or (item.get("tenant_id") is not None and item["tenant_id"] != tenant_id)
            ):
                raise AppError(
                    code=ErrorCode.RETRIEVAL_FAILED,
                    internal_message="candidate identity or scope cannot be verified",
                )

    @classmethod
    def _filter_scope_candidates(cls, chunks):
        output, reasons, seen_ids, seen_content = [], {}, set(), set()
        for item in chunks:
            identity = (item.get("kb_id"), item.get("chunk_id"))
            content_key = (item.get("kb_id"), normalize_content(str(item.get("content") or "")))
            reason = None
            if identity in seen_ids:
                reason = "duplicate_chunk_id"
            elif item.get("index_version") != "v1":
                if (item.get("metadata") or {}).get("chunk_type") == "reference_pointer":
                    reason = "reference_pointer"
                elif is_low_information_content(str(item.get("content") or "")):
                    reason = "low_information"
                elif content_key in seen_content:
                    reason = "duplicate_content"
            if reason:
                reasons[reason] = reasons.get(reason, 0) + 1
            else:
                output.append(item)
                seen_ids.add(identity)
                seen_content.add(content_key)
        return output, {
            "candidate_count_before_filter": len(chunks),
            "candidate_count_after_filter": len(output),
            "filtered_count": len(chunks) - len(output),
            "filtered_reasons": reasons,
        }

    async def _load_knowledge_bases(
        self,
        *,
        kb_ids: list[str],
        tenant_id: str,
    ) -> list[Any]:
        get_by_ids = getattr(self.knowledge_base_repository, "get_by_ids", None)
        if callable(get_by_ids):
            loaded = await get_by_ids(kb_ids=kb_ids, tenant_id=tenant_id)
        else:
            loaded = [
                await self.knowledge_base_repository.get_by_id(kb_id=kb_id, tenant_id=tenant_id)
                for kb_id in kb_ids
            ]

        by_id = {
            str(knowledge_base.id): knowledge_base
            for knowledge_base in (loaded or [])
            if knowledge_base is not None
        }
        for kb_id in kb_ids:
            if kb_id not in by_id:
                raise KnowledgeBaseNotFoundError(missing_kb_id=kb_id)
        return [by_id[kb_id] for kb_id in kb_ids]

    async def _resolve_index_version(
        self,
        requested_version: str | None,
        *,
        knowledge_base: Any,
        tenant_id: str,
    ) -> str:
        version = str(
            requested_version or getattr(knowledge_base, "active_index_version", None) or "v1"
        )
        if requested_version is None:
            return version
        repository = self.index_version_repository
        get_version = getattr(repository, "get", None)
        if not callable(get_version):
            return version
        item = await get_version(
            tenant_id=tenant_id,
            kb_id=str(knowledge_base.id),
            version=version,
        )
        if item is None or str(getattr(item, "status", "")) not in {
            "active",
            "ready",
            "archived",
        }:
            raise_app_error(
                ErrorCode.PARAM_ERROR,
                "index version is not queryable",
                context={"kb_id": str(knowledge_base.id), "index_version": version},
            )
        return version

    @staticmethod
    def _merge_graph_candidates(
        seeds: list[dict[str, Any]], injected: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        existing_ids = {(item.get("kb_id"), str(item.get("chunk_id"))) for item in seeds}
        return [
            *seeds,
            *[
                item
                for item in injected
                if (item.get("kb_id"), str(item.get("chunk_id"))) not in existing_ids
            ],
        ]

    @staticmethod
    def _select_without_rerank(
        *,
        seeds: list[dict[str, Any]],
        injected: list[dict[str, Any]],
        top_k: int,
    ) -> tuple[list[dict[str, Any]], int]:
        if top_k <= 0:
            return [], 0
        quota = min(10, top_k // 5)
        if quota == 0:
            return seeds[:top_k], 0
        seed_ids = {(item.get("kb_id"), str(item.get("chunk_id") or "")) for item in seeds}
        seen_chunk_ids = set(seed_ids)
        selected_anchor_ids: set[str] = set()
        eligible: list[dict[str, Any]] = []
        for item in sorted(
            injected,
            key=lambda candidate: (
                int(candidate.get("_graph_anchor_rank", 999999)),
                int(
                    candidate.get(
                        "_graph_relation_rank",
                        {"reference": 0, "parent_section": 1, "previous": 2, "next": 3}.get(
                            str(
                                candidate.get("injection_source")
                                or (candidate.get("metadata") or {}).get("injection_source")
                                or ""
                            ),
                            99,
                        ),
                    )
                ),
                str(candidate.get("kb_id") or ""),
                str(candidate.get("chunk_id") or ""),
            ),
        ):
            chunk_id = str(item.get("chunk_id") or "")
            anchor_id = str(
                item.get("_graph_anchor_id")
                or (item.get("metadata") or {}).get("injection_anchor_id")
                or ""
            )
            identity = (item.get("kb_id"), chunk_id)
            anchor_identity = (item.get("kb_id"), anchor_id)
            if not chunk_id or identity in seen_chunk_ids or anchor_identity in selected_anchor_ids:
                continue
            seen_chunk_ids.add(identity)
            selected_anchor_ids.add(anchor_identity)
            eligible.append(item)
            if len(eligible) >= quota:
                break
        direct_count = max(0, top_k - len(eligible))
        return [*seeds[:direct_count], *eligible], len(eligible)

    def _validate_query(self, query: str) -> None:
        normalized_query = query.strip()
        if not normalized_query:
            raise_app_error(ErrorCode.PARAM_ERROR, "query must not be blank")

        max_length = self.settings.query_max_length
        if len(normalized_query) > max_length:
            raise_app_error(
                ErrorCode.PARAM_ERROR,
                f"query must not exceed {max_length} characters",
                data={"max_length": max_length},
            )

    async def _resolve_empty_reason(
        self,
        *,
        kb_ids: list[str],
        tenant_id: str,
    ) -> str:
        count_indexed_chunks = getattr(
            self.knowledge_base_repository,
            "count_indexed_chunks",
            None,
        )
        if not callable(count_indexed_chunks):
            return "no_chunks_matched"

        try:
            counts = [
                await count_indexed_chunks(kb_id=kb_id, tenant_id=tenant_id) for kb_id in kb_ids
            ]
        except Exception as exception:
            self.logger.warning(
                "EMPTY_REASON_RESOLUTION_FAILED | tenant_id=%s | kb_ids=%s | error=%s",
                tenant_id,
                kb_ids,
                str(exception) or type(exception).__name__,
            )
            return "no_chunks_matched"

        return (
            "no_indexed_chunks" if sum(int(count) for count in counts) == 0 else "no_chunks_matched"
        )

    @staticmethod
    def _resolve_kb_ids(request: RagRetrieveRequest) -> list[str]:
        if request.kb_ids is not None:
            kb_ids = list(request.kb_ids)
        elif request.kb_id is not None:
            kb_ids = [request.kb_id]
        else:
            raise_app_error(
                ErrorCode.PARAM_ERROR,
                "either kb_id or kb_ids must be provided",
            )

        if not kb_ids:
            raise_app_error(ErrorCode.PARAM_ERROR, "kb_ids must not be empty")
        if len(kb_ids) > MULTI_KB_MAX:
            raise_app_error(
                ErrorCode.PARAM_ERROR,
                f"kb_ids must not contain more than {MULTI_KB_MAX} knowledge bases",
                data={"max_kb": MULTI_KB_MAX},
            )
        return kb_ids

    def _enforce_multi_kb_limit(
        self,
        plan: PlanContext,
        *,
        profile: str,
        kb_ids: list[str],
    ) -> None:
        max_kb_per_retrieve = getattr(plan.limits, "max_kb_per_retrieve", 1)
        if len(kb_ids) <= max_kb_per_retrieve:
            return
        self._raise_feature_not_allowed(
            plan,
            profile=profile,
            feature="multi-knowledge-base retrieval",
            data={
                "plan": plan.plan,
                "profile": profile,
                "kb_count": len(kb_ids),
                "max_kb_per_retrieve": max_kb_per_retrieve,
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

    @staticmethod
    def _expand_profile(
        request: RagRetrieveRequest,
        profile: str,
    ) -> RagRetrieveRequest:
        if profile == "custom":
            return request
        from app.schemas.rerank import RerankOptions

        preset = expand_retrieve_profile(profile)
        conflicts = sorted(
            request.model_fields_set
            & {"top_k", "retrieval_options", "rerank_options", "query_options"}
        )
        if conflicts:
            raise_app_error(
                ErrorCode.PARAM_ERROR,
                "named profiles cannot override advanced parameters",
                data={"profile": profile, "conflicting_fields": conflicts},
            )
        return request.model_copy(
            update={
                "profile": profile,
                "top_k": preset["top_k"],
                "retrieval_options": RetrievalOptions.model_validate(preset["retrieval_options"]),
                "rerank_options": RerankOptions.model_validate(preset["rerank_options"]),
                "query_options": QueryOptions.model_validate(preset["query_options"]),
            }
        )

    def _enforce_plan_features(
        self,
        plan: PlanContext,
        *,
        profile: str,
        mode: RetrievalMode,
        rerank_enabled: bool,
        query_rewrite_enabled: bool,
        evidence_enabled: bool,
    ) -> None:
        if mode == "hybrid" and not plan.features.hybrid_allowed:
            self._raise_feature_not_allowed(plan, profile=profile, feature="hybrid retrieval")
        if rerank_enabled and not plan.features.rerank_allowed:
            self._raise_feature_not_allowed(plan, profile=profile, feature="rerank")
        if query_rewrite_enabled and not plan.features.query_rewrite_allowed:
            self._raise_feature_not_allowed(plan, profile=profile, feature="query rewrite")
        if evidence_enabled and not plan.features.evidence_allowed:
            self._raise_feature_not_allowed(plan, profile=profile, feature="evidence")

    @staticmethod
    def _raise_feature_not_allowed(
        plan: PlanContext,
        *,
        profile: str,
        feature: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        raise_app_error(
            ErrorCode.FEATURE_NOT_ALLOWED,
            f"{feature} is not allowed for the {plan.plan} plan",
            data=data or {"plan": plan.plan, "profile": profile, "feature": feature},
        )

    def _resolve_retrieval_mode(self, request: RagRetrieveRequest) -> RetrievalMode:
        requested_mode = (
            request.retrieval_options.mode if request.retrieval_options is not None else None
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
        if request.retrieval_options is not None:
            options = request.retrieval_options
            illegal = {"vector": "bm25_top_k", "bm25": "vector_top_k"}.get(mode)
            if illegal and illegal in options.model_fields_set:
                raise_app_error(ErrorCode.PARAM_ERROR, f"{illegal} is not valid for {mode}")
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
                else self.settings.hybrid_vector_top_k
            )
            bm25_top_k = (
                options.bm25_top_k
                if options is not None and options.bm25_top_k is not None
                else self.settings.hybrid_bm25_top_k
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

        rrf_k = self.settings.hybrid_rrf_k
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
        empty_reason: str | None = None,
        multi_kb: bool = False,
        kb_count: int | None = None,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "mode": mode,
            "fusion": "rrf" if mode == "hybrid" else "none",
            "rrf_k": rrf_k if mode == "hybrid" else None,
            "vector_store": self.settings.vector_store,
            "keyword_search": (
                self.settings.keyword_search_provider if mode in {"bm25", "hybrid"} else None
            ),
            "vector_top_k": vector_top_k,
            "bm25_top_k": bm25_top_k,
            "vector_count": vector_count,
            "bm25_count": bm25_count,
            "fused_count": fused_count,
        }
        if multi_kb:
            metadata.update(
                {
                    "multi_kb": True,
                    "kb_count": kb_count,
                    "fusion": "rrf",
                    "rrf_k": rrf_k,
                }
            )
        if empty_reason is not None:
            metadata["empty_reason"] = empty_reason
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
        if options.enabled is not None:
            return options.enabled
        if options.strategy == "noop":
            return False
        if options.strategy == "rewrite":
            return True
        return self.settings.query_rewrite_enabled

    def _resolve_evidence_options(self, request: RagRetrieveRequest) -> EvidenceOptions:
        requested = request.evidence_options
        enabled = (
            requested.enabled
            if requested is not None and requested.enabled is not None
            else self.settings.evidence_enabled
        )
        return EvidenceOptions(
            enabled=bool(enabled),
            max_items=requested.max_items if requested is not None else None,
        )

    async def _orchestrate_evidence(
        self,
        *,
        query: str,
        tenant_id: str,
        kb_ids: list[str],
        index_versions: dict[str, str],
        candidates: list[dict[str, Any]],
        retrieved_chunks: list[dict[str, Any]],
        options: EvidenceOptions,
    ) -> EvidenceOrchestrationResult:
        if not options.enabled:
            return EvidenceOrchestrationResult(
                pack=None,
                metadata=RetrieveEvidenceMetadata(enabled=False),
            )
        if self.evidence_orchestration_service is None:
            return EvidenceOrchestrationResult(
                pack=None,
                metadata=RetrieveEvidenceMetadata(
                    enabled=True,
                    degraded=True,
                    error="evidence service unavailable",
                ),
            )
        return await self.evidence_orchestration_service.orchestrate(
            query=query,
            tenant_id=tenant_id,
            kb_ids=kb_ids,
            index_versions=index_versions,
            candidates=candidates,
            retrieved_chunks=retrieved_chunks,
            options=options,
        )

    def _resolve_rerank_top_n(self, request: RagRetrieveRequest) -> int:
        if request.rerank_options is not None and request.rerank_options.top_n is not None:
            return request.rerank_options.top_n
        mode = self._resolve_retrieval_mode(request)
        return self._resolve_retrieval_options(request, mode)[2]

    def _build_rerank_metadata(
        self,
        *,
        enabled: bool,
        top_n: int,
        candidate_count: int,
        returned_count: int,
        latency_ms: int,
        degraded: bool,
        error: str | None,
    ) -> dict[str, Any]:
        provider = "qwen3.7" if enabled else "noop"
        metadata: dict[str, Any] = {
            "enabled": enabled,
            "provider": provider,
            "model": self.settings.rerank_model if enabled else None,
            "top_n": top_n,
            "candidate_count": candidate_count,
            "returned_count": returned_count,
            "latency_ms": latency_ms,
            "degraded": degraded,
        }
        if degraded:
            metadata["error"] = error or "rerank failed"
        return metadata

    @staticmethod
    def _safe_rerank_error(exception: Exception) -> str:
        # Never surface an arbitrary provider exception: it may contain request headers,
        # a URL with credentials, or candidate text.
        safe_messages = {
            "rerank request timed out",
            "rerank response missing output",
            "rerank response has invalid results count",
            "rerank result must be an object",
            "rerank result has invalid index",
            "rerank result has invalid score",
            "rerank credentials or endpoint not configured",
        }
        message = str(exception)
        if message in safe_messages or re.fullmatch(r"rerank HTTP [1-5][0-9]{2}", message):
            return message
        return f"rerank failed ({type(exception).__name__})"
