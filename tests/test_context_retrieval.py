from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.db.base import Base
from app.models.chunk import Chunk
from app.models.document import Document, DocumentStatus
from app.models.knowledge_base import KnowledgeBase
from app.services.chunk_relation_resolver import ChunkRelationResolver
from app.services.context_expansion_service import ContextExpansionService
from app.services.graph_candidate_expansion_service import GraphCandidateExpansionService
from app.services.rag_service import RagService


def _chunk(
    chunk_id: str,
    order: int,
    *,
    content: str = "退款规则",
    section_id: str = "s",
    parent_section_id: str | None = None,
    chunk_type: str = "prose",
) -> SimpleNamespace:
    return SimpleNamespace(
        id=chunk_id,
        document_id="doc",
        tenant_id="tenant",
        kb_id="kb",
        index_version="v2",
        title="规则",
        content=content,
        retrieval_text="章节：退款\n正文：" + content,
        order_index=order,
        section_id=section_id,
        parent_section_id=parent_section_id,
        chunk_metadata={"chunk_type": chunk_type, "heading_path": "退款/流程"},
    )


class _Result:
    def __init__(self, *, relations=(), chunks=()):
        self.relations = relations
        self.chunks = chunks

    def all(self):
        return self.relations

    def scalars(self):
        return SimpleNamespace(
            all=lambda: self.chunks,
            first=lambda: self.chunks[0] if self.chunks else None,
        )


@pytest.mark.asyncio
async def test_v1_never_queries_relations() -> None:
    session = SimpleNamespace(execute=AsyncMock(side_effect=AssertionError("no relation query")))
    anchor = {"chunk_id": "a", "content": "原文", "index_version": "v1", "kb_id": "kb"}
    graph = await GraphCandidateExpansionService(session).expand_candidates(
        tenant_id="tenant", kb_ids=["kb"], seeds=[anchor]
    )
    context = await ContextExpansionService(session).expand(
        tenant_id="tenant", kb_ids=["kb"], anchors=[anchor]
    )
    assert graph.injected_count == 0
    assert context.anchors == [anchor]
    session.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_graph_injects_only_one_hop_and_keeps_original_seed() -> None:
    target = _chunk("ref", 10, content="明确引用的规则")
    session = SimpleNamespace(
        execute=AsyncMock(
                side_effect=[
                    _Result(relations=[(SimpleNamespace(), target)]),
                    _Result(chunks=[]),
                ]
        )
    )
    seed = {
        "chunk_id": "anchor",
        "kb_id": "kb",
        "index_version": "v2",
        "document_id": "doc",
        "order_index": 1,
        "section_id": "s",
        "metadata": {"chunk_type": "prose"},
    }
    result = await GraphCandidateExpansionService(session).expand_candidates(
        tenant_id="tenant",
        kb_ids=["kb"],
        seeds=[seed],
    )
    assert result.injected_count == 1
    assert result.candidates[0]["chunk_id"] == "ref"
    assert result.candidates[0]["kb_id"] == "kb"
    assert result.candidates[0]["injection_source"] == "reference"
    assert session.execute.await_count == 2  # reference target is never expanded recursively


@pytest.mark.asyncio
async def test_graph_query_failure_discards_partial_injection_without_hiding_seed() -> None:
    session = SimpleNamespace(execute=AsyncMock(side_effect=TimeoutError("graph unavailable")))
    seed = {"chunk_id": "anchor", "kb_id": "kb", "index_version": "v2"}
    result = await GraphCandidateExpansionService(session).expand_candidates(
        tenant_id="tenant", kb_ids=["kb"], seeds=[seed]
    )
    assert result.degraded
    assert result.error == "TimeoutError"
    assert result.candidates == []
    assert result.injected_count == 0
    assert seed == {"chunk_id": "anchor", "kb_id": "kb", "index_version": "v2"}


@pytest.mark.asyncio
async def test_context_expansion_preserves_anchor_and_budget_and_degrades() -> None:
    target = _chunk("ref", 3, content="引用说明")
    previous = _chunk("prev", 0, content="前置条件")
    session = SimpleNamespace(
        execute=AsyncMock(
                side_effect=[
                    _Result(relations=[(SimpleNamespace(), target)]),
                    _Result(chunks=[previous]),
                ]
        )
    )
    anchor = {
        "chunk_id": "a",
        "kb_id": "kb",
        "index_version": "v2",
        "document_id": "doc",
        "title": "退款规则.md",
        "content": "原文",
        "score": 0.75,
        "order_index": 1,
        "section_id": "s",
        "metadata": {"chunk_type": "prose"},
    }
    result = await ContextExpansionService(session).expand(
        tenant_id="tenant", kb_ids=["kb"], anchors=[anchor]
    )
    assert [source["relation"] for source in result.anchors[0]["context"]["sources"]] == [
        "anchor",
        "reference",
        "previous",
    ]
    assert result.anchors[0]["score"] == 0.75
    assert result.anchors[0]["context"]["content"] == (
        "文档：退款规则.md\n\n前置条件\n\n原文\n\n引用内容：\n引用说明"
    )
    assert anchor.get("context") is None
    failing = SimpleNamespace(execute=AsyncMock(side_effect=TimeoutError("relations unavailable")))
    degraded = await ContextExpansionService(failing).expand(
        tenant_id="tenant", kb_ids=["kb"], anchors=[anchor]
    )
    assert degraded.degraded and degraded.anchors[0]["context"] is None
    assert degraded.anchors[0]["content"] == "原文"


def test_filter_drops_reference_pointer_but_keeps_short_evidence() -> None:
    chunks = [
        {"chunk_id": "a", "content": "待发货可退款。", "metadata": {}},
        {"chunk_id": "b", "content": "待发货可退款。", "metadata": {}},
        {
            "chunk_id": "c",
            "content": "参见《规则》",
            "metadata": {"chunk_type": "reference_pointer"},
        },
    ]
    filtered, stats = RagService._filter_candidates(chunks)
    assert [chunk["chunk_id"] for chunk in filtered] == ["a"]
    assert stats["filtered_reasons"] == {"duplicate_content": 1, "reference_pointer": 1}


def test_no_rerank_graph_quota_preserves_seed_order_and_scores() -> None:
    seeds = [
        {"chunk_id": f"seed-{index}", "score": 1 - index / 100, "retrieval_source": "hybrid"}
        for index in range(12)
    ]
    injected = [
        {
            "chunk_id": "graph-reference",
            "score": 0.0,
            "vector_score": None,
            "bm25_score": None,
            "retrieval_source": "graph",
            "order_index": 8,
            "_graph_relation_rank": 0,
            "_graph_anchor_rank": 1,
            "_graph_anchor_id": "seed-1",
        },
        {
            "chunk_id": "graph-parent",
            "score": 0.0,
            "vector_score": None,
            "bm25_score": None,
            "retrieval_source": "graph",
            "order_index": 2,
            "_graph_relation_rank": 1,
            "_graph_anchor_rank": 0,
            "_graph_anchor_id": "seed-0",
        },
        {
            "chunk_id": "graph-same-anchor",
            "retrieval_source": "graph",
            "_graph_relation_rank": 2,
            "_graph_anchor_rank": 0,
            "_graph_anchor_id": "seed-0",
        },
    ]
    selected, graph_count = RagService._select_without_rerank(
        seeds=seeds, injected=injected, top_k=10
    )
    assert [item["chunk_id"] for item in selected[:8]] == [
        f"seed-{index}" for index in range(8)
    ]
    assert [item["chunk_id"] for item in selected[8:]] == [
        "graph-reference",
        "graph-parent",
    ]
    assert graph_count == 2
    assert selected[8]["score"] == 0.0
    assert selected[8]["vector_score"] is None
    assert selected[8]["bm25_score"] is None


def test_no_rerank_graph_quota_edges_and_candidate_shortage() -> None:
    seeds = [{"chunk_id": f"seed-{index}"} for index in range(4)]
    injected = [
        {
            "chunk_id": "graph-1",
            "_graph_relation_rank": 0,
            "_graph_anchor_rank": 0,
            "_graph_anchor_id": "seed-0",
        }
    ]
    one, selected_one = RagService._select_without_rerank(
        seeds=seeds, injected=injected, top_k=1
    )
    assert [item["chunk_id"] for item in one] == ["seed-0"]
    assert selected_one == 0
    four, selected_four = RagService._select_without_rerank(
        seeds=seeds, injected=injected, top_k=4
    )
    assert [item["chunk_id"] for item in four] == ["seed-0", "seed-1", "seed-2", "graph-1"]
    assert selected_four == 1


def test_context_natural_order_heading_and_non_prose_neighbor_policy() -> None:
    anchor = {
        "chunk_id": "anchor",
        "title": "买家保障.md",
        "content": "当前命中",
        "metadata": {"heading_path": "3. 买家保障服务/3.2 极速退款"},
    }
    supplemental = [
        (_chunk("reference", 9, content="引用正文"), "reference"),
        (_chunk("parent", 1, content="父章节正文"), "parent_section"),
        (_chunk("previous", 2, content="前一片段"), "previous"),
        (_chunk("next", 4, content="后一片段"), "next"),
    ]
    context = ContextExpansionService._build_context(anchor, supplemental)
    assert context["content"] == (
        "文档：买家保障.md\n\n章节：3. 买家保障服务/3.2 极速退款\n\n"
        "父章节正文\n\n前一片段\n\n当前命中\n\n后一片段\n\n"
        "引用内容：\n引用正文"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("chunk_type", ["table", "list", "code"])
async def test_non_prose_chunks_do_not_query_same_section_neighbors(chunk_type: str) -> None:
    session = SimpleNamespace(
        execute=AsyncMock(
            side_effect=[
                _Result(relations=[]),
                _Result(chunks=[]),
            ]
        )
    )
    anchor = {
        "chunk_id": "anchor",
        "kb_id": "kb",
        "index_version": "v2",
        "document_id": "doc",
        "parent_section_id": "parent",
        "section_id": "section",
        "order_index": 5,
        "metadata": {"chunk_type": chunk_type},
    }
    related = await ChunkRelationResolver(session).resolve(
        tenant_id="tenant", kb_id="kb", index_version="v2", anchor=anchor
    )
    assert related == []
    assert session.execute.await_count == 2


@pytest.mark.asyncio
async def test_relation_resolver_never_uses_sibling_as_parent() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    async with sessions() as session:
        session.add(KnowledgeBase(id="kb", tenant_id="tenant", name="测试库"))
        session.add(
            Document(
                id="doc",
                tenant_id="tenant",
                kb_id="kb",
                title="买家保障.md",
                status=int(DocumentStatus.SUCCESS),
            )
        )

        def make_chunk(
            chunk_id: str,
            content: str,
            section_id: str,
            parent_section_id: str,
            order_index: int,
        ) -> Chunk:
            return Chunk(
                id=chunk_id,
                tenant_id="tenant",
                kb_id="kb",
                document_id="doc",
                title="买家保障.md",
                content=content,
                retrieval_text=content,
                index_version="v2",
                section_id=section_id,
                parent_section_id=parent_section_id,
                order_index=order_index,
                chunk_metadata={"chunk_type": "prose"},
                embedding=[0.0] * 1536,
            )

        session.add_all(
            [
                make_chunk("parent-old", "父章节较早正文", "section-3", "root", 0),
                make_chunk("parent-near", "父章节最近正文", "section-3", "root", 1),
                make_chunk("sibling", "3.1 正品保障", "section-3-1", "section-3", 2),
                make_chunk("anchor", "3.2 极速退款", "section-3-2", "section-3", 3),
            ]
        )
        await session.commit()
        related = await ChunkRelationResolver(session).resolve(
            tenant_id="tenant",
            kb_id="kb",
            index_version="v2",
            anchor={
                "chunk_id": "anchor",
                "document_id": "doc",
                "section_id": "section-3-2",
                "parent_section_id": "section-3",
                "order_index": 3,
                "metadata": {"chunk_type": "prose"},
            },
        )
        parents = [item.chunk.id for item in related if item.relation == "parent_section"]
        assert parents == ["parent-near"]
        assert all(item.chunk.id != "sibling" for item in related)
    await engine.dispose()


@pytest.mark.asyncio
async def test_relation_resolver_same_section_neighbors_are_strictly_isolated() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    async with sessions() as session:
        for kb_id, tenant_id in (("kb", "tenant"), ("other-kb", "tenant")):
            session.add(KnowledgeBase(id=kb_id, tenant_id=tenant_id, name=kb_id))
        for document_id, kb_id in (("doc", "kb"), ("other-doc", "kb"), ("cross-kb", "other-kb")):
            session.add(
                Document(
                    id=document_id,
                    tenant_id="tenant",
                    kb_id=kb_id,
                    title=document_id,
                    status=int(DocumentStatus.SUCCESS),
                )
            )

        def add_chunk(
            chunk_id: str,
            *,
            document_id: str = "doc",
            kb_id: str = "kb",
            version: str = "v2",
            section_id: str = "section-3-2",
            order_index: int,
        ) -> None:
            session.add(
                Chunk(
                    id=chunk_id,
                    tenant_id="tenant",
                    kb_id=kb_id,
                    document_id=document_id,
                    title=document_id,
                    content=chunk_id,
                    retrieval_text=chunk_id,
                    index_version=version,
                    section_id=section_id,
                    parent_section_id="section-3",
                    order_index=order_index,
                    chunk_metadata={"chunk_type": "prose"},
                    embedding=[0.0] * 1536,
                )
            )

        add_chunk("anchor", order_index=5)
        add_chunk("next", order_index=6)
        add_chunk("sibling", section_id="section-3-3", order_index=4)
        add_chunk("other-version", version="v1", order_index=4)
        add_chunk("other-document", document_id="other-doc", order_index=4)
        add_chunk("other-kb-chunk", document_id="cross-kb", kb_id="other-kb", order_index=4)
        await session.commit()
        related = await ChunkRelationResolver(session).resolve(
            tenant_id="tenant",
            kb_id="kb",
            index_version="v2",
            anchor={
                "chunk_id": "anchor",
                "document_id": "doc",
                "section_id": "section-3-2",
                "order_index": 5,
                "metadata": {"chunk_type": "prose"},
            },
        )
        assert [(item.chunk.id, item.relation) for item in related] == [("next", "next")]
    await engine.dispose()
