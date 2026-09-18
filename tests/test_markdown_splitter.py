from app.utils.markdown_splitter import MarkdownStructuredSplitter


def test_heading_boundaries_and_heading_paths() -> None:
    splitter = MarkdownStructuredSplitter(chunk_size=200, chunk_overlap=20)

    pieces = splitter.split(
        "# 平台交易规则\n\n"
        "交易规则总览。\n\n"
        "## 退款时效\n\n"
        "退款会在审核通过后处理。\n\n"
        "### 特殊情况\n\n"
        "节假日可能顺延。\n\n"
        "## 订单取消\n\n"
        "下单后可以在发货前取消。\n\n"
        "---\n"
    )

    assert [piece.metadata["heading_path"] for piece in pieces] == [
        "平台交易规则",
        "平台交易规则/退款时效",
        "平台交易规则/订单取消",
    ]
    assert "### 特殊情况" in pieces[1].text
    assert "节假日可能顺延。" in pieces[1].text
    assert all("\n---\n" not in f"\n{piece.text}\n" for piece in pieces)


def test_long_section_splits_inside_section_and_keeps_heading_path() -> None:
    splitter = MarkdownStructuredSplitter(chunk_size=50, chunk_overlap=8)

    pieces = splitter.split(
        "# 产品说明\n\n"
        "第一段内容用于测试章节内部的二次切分，不应该影响章节元数据。\n\n"
        "第二段内容继续补充产品说明和使用边界。"
    )

    assert len(pieces) > 1
    assert {piece.metadata["heading_path"] for piece in pieces} == {"产品说明"}
    assert all(piece.metadata["chunk_type"] == "section" for piece in pieces)
    assert pieces[0].text.startswith("# 产品说明")


def test_code_fence_stays_in_one_section_piece() -> None:
    splitter = MarkdownStructuredSplitter(chunk_size=100, chunk_overlap=10)

    pieces = splitter.split(
        "# 示例\n\n"
        "```python\n"
        "def answer():\n"
        "    return 'ok'\n"
        "```\n\n"
        "代码说明。"
    )

    assert len(pieces) == 1
    assert "```python\n" in pieces[0].text
    assert "    return 'ok'\n```" in pieces[0].text


def test_untitled_markdown_falls_back_to_paragraph_splitting() -> None:
    splitter = MarkdownStructuredSplitter(chunk_size=32, chunk_overlap=4)

    pieces = splitter.split(
        "第一段没有标题，但内容足够长，需要在段落或字符边界切开。\n\n"
        "第二段继续提供无标题文档的内容。"
    )

    assert len(pieces) > 1
    assert all(piece.metadata["heading_path"] is None for piece in pieces)
    assert all(piece.metadata["chunk_type"] == "section" for piece in pieces)


def test_small_table_is_one_independent_table_piece() -> None:
    splitter = MarkdownStructuredSplitter(chunk_size=200, chunk_overlap=10)

    pieces = splitter.split(
        "# 规则\n\n"
        "表格前的说明。\n\n"
        "| 场景 | 是否可叠加 | 说明 |\n"
        "| --- | --- | --- |\n"
        "| 满减 + 优惠券 | 否 | 同一订单只能选一种 |\n\n"
        "表格后的说明。"
    )

    assert [piece.metadata["chunk_type"] for piece in pieces] == [
        "section",
        "table",
        "section",
    ]
    table_piece = pieces[1]
    assert table_piece.metadata["heading_path"] == "规则"
    assert table_piece.metadata["table_part"] == 1
    assert "| 场景 | 是否可叠加 | 说明 |" in table_piece.text
    assert "| 满减 + 优惠券 | 否 | 同一订单只能选一种 |" in table_piece.text


def test_large_table_repeats_header_and_increments_table_part() -> None:
    splitter = MarkdownStructuredSplitter(
        chunk_size=60,
        chunk_overlap=10,
        table_max_rows_per_chunk=2,
    )

    pieces = splitter.split(
        "## 兼容性\n\n"
        "| 场景 | 是否可叠加 | 说明 |\n"
        "| --- | --- | --- |\n"
        "| 满减 | 否 | 订单只能选一种优惠 |\n"
        "| 优惠券 | 否 | 订单只能选一种优惠 |\n"
        "| 积分 | 是 | 可以与基础优惠同时使用 |\n"
        "| 会员折扣 | 是 | 会员权益可叠加 |\n"
        "| 包邮 | 是 | 满足条件后自动生效 |\n"
    )

    table_pieces = [piece for piece in pieces if piece.metadata["chunk_type"] == "table"]
    assert len(table_pieces) == 3
    assert [piece.metadata["table_part"] for piece in table_pieces] == [1, 2, 3]
    assert all("【表头】场景 | 是否可叠加 | 说明" in piece.text for piece in table_pieces)
    assert all(piece.metadata["heading_path"] == "兼容性" for piece in table_pieces)
    assert all("| --- | --- | --- |" not in piece.text for piece in table_pieces)


def test_mixed_document_keeps_tables_separate_from_paragraphs() -> None:
    splitter = MarkdownStructuredSplitter(chunk_size=200, chunk_overlap=10)

    pieces = splitter.split(
        "# 使用说明\n\n"
        "开始说明。\n\n"
        "| 参数 | 值 |\n"
        "| --- | --- |\n"
        "| timeout | 30 |\n\n"
        "结束说明。"
    )

    assert [piece.metadata["chunk_type"] for piece in pieces] == [
        "section",
        "table",
        "section",
    ]
    assert pieces[0].text.endswith("开始说明。")
    assert pieces[2].text == "结束说明。"
