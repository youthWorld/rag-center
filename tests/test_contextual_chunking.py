from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.indexing_service import IndexingService
from app.services.reference_relation_service import ReferenceRelationService
from app.utils.markdown_splitter import MarkdownStructuredSplitter, is_low_information_content
from app.utils.text_splitter import TextSplitter


def test_low_information_and_short_evidence() -> None:
    for content in ("", "---", "***", "！？", "# 标题", "| 表头 |\n| --- |"):
        assert is_low_information_content(content)
    assert not is_low_information_content("待发货可退款。")


def test_v2_section_path_order_table_and_reference_pointer() -> None:
    splitter = MarkdownStructuredSplitter(chunk_size=85, chunk_overlap=0)
    pieces = splitter.split_contextual(
        "# 交易规则\n## 退款规则\n### 未发货\n待发货可退款。\n\n"
        "> 关联规则：参见《优惠券规则》退款处理\n\n"
        "| 订单 | 说明 |\n| --- | --- |\n| A | 保留表头 |\n| B | 保留表头 |",
        document_id="doc",
        index_version="v2",
    )
    assert pieces
    assert [p.metadata["order_index"] for p in pieces] == list(range(len(pieces)))
    assert all(p.metadata["heading_path"] == "交易规则/退款规则/未发货" for p in pieces)
    assert all(p.metadata["parent_section_id"] != p.metadata["section_id"] for p in pieces)
    assert all(p.metadata["section_id"] == pieces[0].metadata["section_id"] for p in pieces)
    assert any("关联规则" in p.text and "待发货可退款" in p.text for p in pieces)
    assert all(p.metadata["chunk_type"] != "reference_pointer" for p in pieces)
    assert any("【表头】" in p.text or "| 订单 |" in p.text for p in pieces)


def test_reference_pointer_before_evidence_is_not_an_independent_chunk() -> None:
    splitter = MarkdownStructuredSplitter(chunk_size=32, chunk_overlap=0)
    pieces = splitter.split_contextual(
        "# 交易规则\n\n> 关联规则：参见《优惠券规则》\n\n待发货可退款。",
        document_id="doc",
        index_version="v2",
    )

    assert len(pieces) == 1
    assert "关联规则" in pieces[0].text
    assert "待发货可退款" in pieces[0].text


@pytest.mark.asyncio
async def test_embedding_uses_retrieval_text_but_content_remains_raw() -> None:
    embedding = SimpleNamespace(embed_documents=AsyncMock(return_value=[[0.1]]))
    vector = SimpleNamespace(add_chunks=AsyncMock())
    service = IndexingService(
        splitter=TextSplitter(chunk_size=100, chunk_overlap=0),
        embedding_provider=embedding,
        vector_store=vector,
    )
    document = SimpleNamespace(
        id="doc",
        tenant_id="tenant",
        kb_id="kb",
        title="规则.md",
        source_type="markdown",
        content="# 退款\n待发货可退款。",
    )
    assert await service.index_document(document, index_version="v2") == 1
    chunk = vector.add_chunks.await_args.args[0][0]
    assert chunk["content"] == "# 退款\n待发货可退款。"
    assert chunk["retrieval_text"] == embedding.embed_documents.await_args.args[0][0]
    assert "文档：规则.md" in chunk["retrieval_text"]
    assert chunk["order_index"] == 0


def test_reference_requires_unambiguous_document_and_heading() -> None:
    source = SimpleNamespace(
        id="s",
        document_id="sdoc",
        section_id="root",
        order_index=0,
        chunk_metadata={"heading_path": "说明"},
    )
    target = SimpleNamespace(
        id="t",
        document_id="tdoc",
        section_id="refund",
        order_index=0,
        chunk_metadata={"heading_path": "优惠券/退款处理"},
    )
    rows = [(source, "说明"), (target, "优惠券规则.md")]
    ref = ReferenceRelationService._parse_target("[详见](优惠券规则.md#退款处理)")
    assert [item[0].id for item in ReferenceRelationService._match_target(rows, ref)] == ["t"]
    other = SimpleNamespace(
        id="u",
        document_id="other",
        section_id="refund",
        order_index=0,
        chunk_metadata={"heading_path": "退款处理"},
    )
    assert len(ReferenceRelationService._match_target(rows + [(other, "优惠券规则")], ref)) == 2
