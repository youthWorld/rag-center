import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.core.error_codes import ErrorCode
from app.core.exceptions import AppError
from app.providers.query.base import QueryProcessResult
from app.schemas.rag import RagRetrieveRequest
from app.services.graph_candidate_expansion_service import GraphCandidateExpansionResult
from app.services.hybrid_search_service import HybridSearchService
from app.services.rag_service import RagService
from app.services.retrieve_once_service import RetrieveOnceService
from app.tenant.plan_resolver import resolve_plan


def chunk(i, kb="a", version="v2", **extra):
    return {
        "chunk_id": str(i),
        "document_id": "doc",
        "kb_id": kb,
        "index_version": version,
        "tenant_id": "tenant",
        "title": "rules",
        "content": f"Rule number {i}: refund within seven business days.",
        "score": 1 / (int(i) + 1),
        **extra,
    }


def service(vector=None, keyword=None):
    bases = {kb: SimpleNamespace(id=kb, name=kb, active_index_version="v2") for kb in ("a", "b")}
    return RagService(
        session=SimpleNamespace(),
        settings=Settings(hybrid_rrf_k=71),
        knowledge_base_repository=SimpleNamespace(
            get_by_ids=AsyncMock(side_effect=lambda kb_ids, tenant_id: [bases[k] for k in kb_ids]),
        ),
        embedding_provider=SimpleNamespace(embed_query=AsyncMock(return_value=[1.0])),
        vector_store=SimpleNamespace(similarity_search_scope=AsyncMock(return_value=vector or [])),
        keyword_search_provider=SimpleNamespace(
            keyword_search_scope=AsyncMock(return_value=keyword or [])
        ),
        graph_candidate_expansion_service=SimpleNamespace(
            expand_candidates=AsyncMock(return_value=GraphCandidateExpansionResult())
        ),
        context_expansion_service=SimpleNamespace(
            expand=AsyncMock(
                side_effect=lambda **kw: SimpleNamespace(
                    anchors=kw["anchors"],
                    expanded_anchor_count=0,
                    supplemental_chunk_count=0,
                    deduplicated_count=0,
                    latency_ms=0,
                    degraded=False,
                    error=None,
                )
            )
        ),
    )


def request(**kw):
    return RagRetrieveRequest(kb_ids=["a", "b"], user_id="u", query="refund", **kw)


@pytest.mark.asyncio
async def test_global_scope_has_one_call_and_no_library_quota():
    svc = service([chunk(i) for i in range(12)], [chunk(i) for i in range(10)])
    result = await svc.retrieve(request(profile="balanced"), tenant_id="tenant")
    assert len(result.retrieved_chunks) == 5
    assert {c.kb_id for c in result.retrieved_chunks} == {"a"}
    svc.embedding_provider.embed_query.assert_awaited_once_with("refund")
    svc.vector_store.similarity_search_scope.assert_awaited_once_with(
        [1.0], tenant_id="tenant", index_versions={"a": "v2", "b": "v2"}, top_k=10
    )
    svc.keyword_search_provider.keyword_search_scope.assert_awaited_once_with(
        query="refund", tenant_id="tenant", index_versions={"a": "v2", "b": "v2"}, top_k=10
    )
    assert result.metadata["application_model_calls"] == 1
    assert result.metadata["retrieval"]["rrf_k"] == 71
    assert result.metadata["retrieval"]["source_distribution"] == {"a": 5, "b": 0}


@pytest.mark.asyncio
async def test_embedding_and_bm25_start_before_either_completes():
    svc = service()
    embedding_started, bm25_started, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def embed(query):
        embedding_started.set()
        await release.wait()
        return [1.0]

    async def bm25(**kwargs):
        bm25_started.set()
        await release.wait()
        return [chunk(1)]

    svc.embedding_provider.embed_query.side_effect = embed
    svc.keyword_search_provider.keyword_search_scope.side_effect = bm25
    task = asyncio.create_task(svc.retrieve(request(), tenant_id="tenant"))
    try:
        await asyncio.wait_for(asyncio.gather(embedding_started.wait(), bm25_started.wait()), 1)
        svc.vector_store.similarity_search_scope.assert_not_awaited()
        release.set()
        result = await task
        metrics = result.metadata["retrieval"]
        assert all(
            key in metrics
            for key in (
                "embedding_latency_ms",
                "vector_store_latency_ms",
                "vector_latency_ms",
                "bm25_latency_ms",
                "hybrid_latency_ms",
            )
        )
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancellation_propagates_to_both_branches():
    svc = service()
    started = [asyncio.Event(), asyncio.Event()]
    stopped = [asyncio.Event(), asyncio.Event()]

    async def wait(index):
        started[index].set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped[index].set()

    svc.embedding_provider.embed_query.side_effect = lambda query: None

    async def embedding(query):
        return await wait(0)

    async def keyword(**kwargs):
        return await wait(1)

    svc.embedding_provider.embed_query.side_effect = embedding
    svc.keyword_search_provider.keyword_search_scope.side_effect = keyword
    task = asyncio.create_task(svc.retrieve(request(), tenant_id="tenant"))
    await asyncio.wait_for(asyncio.gather(*(event.wait() for event in started)), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert all(event.is_set() for event in stopped)


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", ["vector", "embedding", "bm25", "both"])
async def test_branch_failures_and_actual_counts(failed):
    svc = service([chunk(1)], [chunk(2)])
    if failed in {"embedding", "both"}:
        svc.embedding_provider.embed_query.side_effect = RuntimeError("failed")
    if failed == "vector":
        svc.vector_store.similarity_search_scope.side_effect = RuntimeError("failed")
    if failed in {"bm25", "both"}:
        svc.keyword_search_provider.keyword_search_scope.side_effect = RuntimeError("failed")
    if failed == "both":
        with pytest.raises(AppError) as error:
            await svc.retrieve(request(), tenant_id="tenant")
        assert error.value.code == 20003
        assert error.value.context["retrieval"]["vector_request_count"] == 0
        return
    result = await svc.retrieve(request(), tenant_id="tenant")
    metadata = result.metadata["retrieval"]
    assert metadata["degraded"]
    assert metadata["fusion"] == "none"
    assert metadata["embedding_request_count"] == 1
    assert metadata["vector_request_count"] == int(failed != "embedding")
    assert result.metadata["application_model_calls"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation",
    [
        {"kb_id": "other"},
        {"index_version": "v1"},
        {"tenant_id": "other"},
        {"document_id": ""},
        {"chunk_id": None},
    ],
)
async def test_invalid_result_scope_fails_entire_retrieval(mutation):
    svc = service([{**chunk(1), **mutation}], [chunk(2)])
    with pytest.raises(AppError) as raised:
        await svc.retrieve(request(), tenant_id="tenant")
    assert raised.value.code == ErrorCode.RETRIEVAL_FAILED.code


def test_global_rrf_identity_duplicates_and_ties():
    svc = HybridSearchService(rrf_k=71)
    a, b = chunk(1), chunk(1, kb="b")
    result = svc.fuse([a, a, b], [b, a, a])
    assert len(result) == 2
    assert result[0]["kb_id"] == "a"
    assert result[0]["score"] == pytest.approx(1 / 72 + 1 / 73)
    assert result[1]["score"] == pytest.approx(1 / 74 + 1 / 72)
    tied = svc.fuse([a], [b])
    assert [row["kb_id"] for row in tied] == ["a", "b"]


@pytest.mark.asyncio
async def test_quality_20_plus_10_and_base_only_failure_fallback():
    svc = service([chunk(i) for i in range(20)], [chunk(i) for i in range(20, 40)])
    graph = [chunk(i, retrieval_source="graph") for i in range(40, 50)]
    svc.graph_candidate_expansion_service.expand_candidates.return_value = (
        GraphCandidateExpansionResult(candidates=graph, injected_count=10)
    )
    svc.query_pipeline = SimpleNamespace(
        process=AsyncMock(
            return_value=QueryProcessResult(
                raw_query="refund",
                effective_query="rewritten",
                search_query="rewritten",
                strategy="rewrite",
                application_model_calls=1,
            )
        )
    )

    async def rerank(**kwargs):
        assert kwargs["query"] == "rewritten"
        assert len(kwargs["chunks"]) == 30
        assert kwargs["top_n"] == 5
        return list(reversed(kwargs["chunks"]))[:5]

    svc.rerank_provider = SimpleNamespace(rerank=AsyncMock(side_effect=rerank))
    result = await svc.retrieve(request(profile="quality"), tenant_id="tenant")
    assert len(result.retrieved_chunks) == 5
    assert all(c.retrieval_source == "graph" for c in result.retrieved_chunks)
    assert result.metadata["retrieval"]["fused_count"] == 40
    assert result.metadata["retrieval"]["base_candidate_count"] == 20
    assert result.metadata["retrieval"]["rerank_input_count"] == 30
    assert result.metadata["application_model_calls"] == 3
    svc.rerank_provider.rerank.side_effect = RuntimeError("private")
    fallback = await svc.retrieve(request(profile="quality"), tenant_id="tenant")
    assert len(fallback.retrieved_chunks) == 5
    assert all(c.retrieval_source != "graph" for c in fallback.retrieved_chunks)
    assert "private" not in fallback.metadata["rerank"]["error"]


@pytest.mark.asyncio
async def test_reference_library_is_sorted_but_response_echo_is_not():
    svc = service([chunk(1)])
    references = []

    async def process(raw, *, knowledge_base, query_options):
        references.append(knowledge_base.id)
        return QueryProcessResult(raw_query=raw, effective_query=raw, search_query=raw)

    svc.query_pipeline = SimpleNamespace(process=process)
    first = await svc.retrieve(request(), tenant_id="tenant")
    second = await svc.retrieve(
        RagRetrieveRequest(kb_ids=["b", "a"], query="refund", user_id="u"), tenant_id="tenant"
    )
    assert references == ["a", "a"]
    assert first.retrieved_chunks == second.retrieved_chunks
    assert first.kb_id == "a" and second.kb_id == "b"


@pytest.mark.asyncio
async def test_research_keeps_top10_and_private_snapshot():
    svc = service([chunk(i) for i in range(20)], [chunk(i) for i in range(20, 40)])
    svc.rerank_provider = SimpleNamespace(
        rerank=AsyncMock(side_effect=lambda **kw: kw["chunks"][: kw["top_n"]])
    )
    result = await RetrieveOnceService(svc).execute(
        tenant_id="tenant",
        user_id="u",
        kb_ids=["a", "b"],
        index_versions={"a": "v2", "b": "v2"},
        query_id="Q1",
        search_query="subquery",
        aspect_ids=["A1"],
        round=1,
        plan=resolve_plan("pro"),
    )
    assert len(result.retrieved_chunks) == 10
    assert len(result.candidate_snapshot) == 40
    assert result.metadata["application_model_calls"] == 2
    assert result.metadata["tenant_policy"]["retrieve_profile"] == "research_fixed"
    assert "_candidate_snapshot" not in result.metadata
    public = await svc.retrieve(request(), tenant_id="tenant")
    assert "_candidate_snapshot" not in public.model_dump_json()


@pytest.mark.parametrize("top_k,quota", [(1, 0), (4, 0), (5, 1), (10, 2), (50, 10)])
def test_public_graph_quota(top_k, quota):
    seeds = [chunk(i) for i in range(50)]
    graph = [
        chunk(i + 50, _graph_anchor_rank=i, _graph_anchor_id=str(i), _graph_relation_rank=0)
        for i in range(12)
    ]
    selected, count = RagService._select_without_rerank(seeds=seeds, injected=graph, top_k=top_k)
    assert len(selected) == top_k
    assert count == quota


@pytest.mark.parametrize(
    "fields",
    [
        {"top_k": 51},
        {"top_k": 0},
        {"retrieval_options": {"vector_top_k": 101}},
        {"rerank_options": {"top_n": 51}},
        {"retrieval_options": {"rrf_k": 60}},
        {"unknown": True},
        {"query_options": {"unknown": True}},
        {"evidence_options": {"unknown": True}},
    ],
)
def test_public_schema_rejects_unsafe_or_unknown_parameters(fields):
    with pytest.raises(ValidationError):
        request(profile="custom", **fields)


@pytest.mark.asyncio
async def test_default_balanced_does_not_bypass_free_plan_and_evidence_is_independent():
    svc = service()
    with pytest.raises(AppError) as error:
        await svc.retrieve(
            RagRetrieveRequest(kb_id="a", user_id="u", query="q", profile=None),
            tenant_id="tenant",
            plan_context=resolve_plan("free"),
        )
    assert error.value.code == ErrorCode.FEATURE_NOT_ALLOWED.code
    for profile in ("speed", "balanced", "quality"):
        req = request(profile=profile, evidence_options={"enabled": True})
        effective = svc._expand_profile(req, profile)
        assert effective.evidence_options.enabled
        assert effective.top_k == 5
        assert not effective.query_options.synonym_enabled


@pytest.mark.asyncio
@pytest.mark.parametrize("kb_ids,snapshot_size", [(["a"], 20), (["a", "b"], 40)])
async def test_evidence_base_snapshot_window_excludes_graph(kb_ids, snapshot_size):
    svc = service([chunk(i) for i in range(20)], [chunk(i) for i in range(20, 40)])
    svc._orchestrate_evidence = AsyncMock(wraps=svc._orchestrate_evidence)
    svc.graph_candidate_expansion_service.expand_candidates.return_value = (
        GraphCandidateExpansionResult(
            candidates=[chunk(100, retrieval_source="graph")], injected_count=1
        )
    )
    svc.rerank_provider = SimpleNamespace(
        rerank=AsyncMock(side_effect=lambda **kw: kw["chunks"][:5])
    )
    response = await svc.retrieve(
        RagRetrieveRequest(kb_ids=kb_ids, user_id="u", query="refund", profile="quality"),
        tenant_id="tenant",
    )
    snapshot = svc._orchestrate_evidence.await_args.kwargs["candidates"]
    assert len(snapshot) == snapshot_size
    assert all(c["chunk_id"] != "100" for c in snapshot)
    assert "_candidate_snapshot" not in response.model_dump_json()
    assert response.metadata["retrieval"]["rerank_input_count"] == 21


@pytest.mark.asyncio
async def test_filtered_shortage_does_not_refill_or_expand_quality_budget():
    svc = service([chunk(1), chunk(2, metadata={"chunk_type": "reference_pointer"})], [])
    svc.rerank_provider = SimpleNamespace(
        rerank=AsyncMock(side_effect=lambda **kw: kw["chunks"][: kw["top_n"]])
    )
    response = await svc.retrieve(request(profile="quality"), tenant_id="tenant")
    assert len(response.retrieved_chunks) == 1
    svc.embedding_provider.embed_query.assert_awaited_once()
    svc.vector_store.similarity_search_scope.assert_awaited_once()
    svc.keyword_search_provider.keyword_search_scope.assert_awaited_once()
    assert (
        svc.graph_candidate_expansion_service.expand_candidates.await_args.kwargs["max_injected"]
        == 10
    )
    assert svc.rerank_provider.rerank.await_args.kwargs["top_n"] == 1
