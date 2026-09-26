from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.core.logging import research_retrieval_log_scope
from app.observability.langfuse_client import RetrieveObservability
from app.schemas.hybrid_search import RetrievalOptions
from app.schemas.rag import EvidenceOptions, QueryOptions, RagRetrieveRequest, RetrievedChunk
from app.schemas.rerank import RerankOptions
from app.services.rag_service import RagService
from app.tenant.plan_resolver import PlanContext


@dataclass(slots=True)
class RetrieveOnceResult:
    query_id: str
    query: str
    aspect_ids: list[str]
    round: int
    candidate_snapshot: list[dict[str, Any]]
    retrieved_chunks: list[RetrievedChunk]
    metadata: dict[str, Any]
    latency_ms: int
    degraded: bool = False
    error: str | None = None


class RetrieveOnceService:
    """Run the shared retrieval kernel without interface-level side effects."""

    def __init__(self, rag_service: RagService) -> None:
        self.rag_service = rag_service

    async def execute(
        self,
        *,
        tenant_id: str,
        user_id: str,
        kb_ids: list[str],
        index_versions: dict[str, str],
        query_id: str,
        search_query: str,
        aspect_ids: list[str],
        round: int,
        plan: PlanContext,
    ) -> RetrieveOnceResult:
        request = RagRetrieveRequest(
            kb_ids=kb_ids,
            user_id=user_id,
            query=search_query,
            profile="custom",
            top_k=20,
            retrieval_options=RetrievalOptions(
                mode="hybrid",
                vector_top_k=20,
                bm25_top_k=20,
                rrf_k=self.rag_service.settings.hybrid_rrf_k,
            ),
            rerank_options=RerankOptions(enabled=True, top_n=10),
            query_options=QueryOptions(
                enabled=False,
                strategy="noop",
                synonym_enabled=False,
            ),
            evidence_options=EvidenceOptions(enabled=False),
            observability_enabled=False,
        )
        observability = RetrieveObservability(
            settings=self.rag_service.settings,
            tenant_id=tenant_id,
            kb_id=kb_ids[0],
            kb_ids=kb_ids,
            user_id=user_id,
            profile="research_fixed",
            plan=plan.plan,
            raw_query=search_query,
            enabled=False,
        )
        with research_retrieval_log_scope():
            response = await self.rag_service._retrieve(
                request,
                kb_ids=kb_ids,
                tenant_id=tenant_id,
                plan=plan,
                profile="research_fixed",
                effective_request=request,
                mode="hybrid",
                rerank_enabled=True,
                query_rewrite_enabled=False,
                evidence_options=EvidenceOptions(enabled=False),
                observability=observability,
                internal_retrieve_once=True,
                frozen_index_versions=index_versions,
            )
        metadata = dict(response.metadata)
        snapshot = metadata.pop("_candidate_snapshot", [])
        retrieval_metadata = metadata.get("retrieval") or {}
        rerank_metadata = metadata.get("rerank") or {}
        degraded = bool(
            retrieval_metadata.get("degraded")
            or rerank_metadata.get("degraded")
            or (metadata.get("context_expansion") or {}).get("degraded")
        )
        return RetrieveOnceResult(
            query_id=query_id,
            query=search_query,
            aspect_ids=list(aspect_ids),
            round=round,
            candidate_snapshot=[dict(item) for item in snapshot],
            retrieved_chunks=response.retrieved_chunks,
            metadata=metadata,
            latency_ms=int(metadata.get("latency_ms") or 0),
            degraded=degraded,
        )
