import zipfile
from pathlib import Path

import pytest

from app.core.error_codes import ErrorCode
from app.core.exceptions import AppError
from app.providers.parsers import (
    DocxParser,
    MarkdownParser,
    NoopParser,
    PdfParser,
    get_document_parser,
    is_supported_document,
)
from app.services.document_ingestion import prepare_document_content
from app.utils.markdown_splitter import MarkdownStructuredSplitter


@pytest.mark.asyncio
async def test_markdown_parser_normalizes_text_and_source_type() -> None:
    parsed = await MarkdownParser().parse(
        b"\xef\xbb\xbf# Heading\n\nBody",
        filename="rules.md",
    )

    assert parsed.content == "# Heading\n\nBody"
    assert parsed.source_type == "markdown"
    assert parsed.metadata["parser"] == "plain_text"


def test_parser_registry_selects_supported_formats_and_noop_fallback() -> None:
    assert is_supported_document("rules.md")
    assert is_supported_document("policy.DOCX")
    assert is_supported_document("guide.pdf")
    assert not is_supported_document("archive.zip")
    assert isinstance(get_document_parser("archive.zip"), NoopParser)


@pytest.mark.asyncio
async def test_prepare_document_content_reads_source_file_once(tmp_path: Path) -> None:
    source = tmp_path / "rules.md"
    source.write_text("# Stored heading\n\nStored content", encoding="utf-8")

    parsed = await prepare_document_content(
        content=None,
        file_path=str(source),
        filename=source.name,
    )

    assert parsed.content == "# Stored heading\n\nStored content"
    assert parsed.source_type == "markdown"


@pytest.mark.asyncio
async def test_prepare_document_content_rejects_empty_content() -> None:
    with pytest.raises(AppError) as raised:
        await prepare_document_content(
            content=" ",
            file_path=None,
            filename="empty.txt",
        )

    assert raised.value.error_code == ErrorCode.DOCUMENT_PARSE_FAILED


@pytest.mark.asyncio
async def test_noop_parser_rejects_unsupported_format_with_param_error() -> None:
    with pytest.raises(AppError) as raised:
        await NoopParser().parse(b"data", filename="archive.zip")

    assert raised.value.error_code == ErrorCode.PARAM_ERROR


@pytest.mark.asyncio
async def test_docx_parser_preserves_heading_levels_and_tables(tmp_path: Path) -> None:
    pytest.importorskip("mammoth")
    source = tmp_path / "fixture.docx"
    _write_docx_fixture(source)

    parsed = await DocxParser().parse(source.read_bytes(), filename=source.name)
    pieces = MarkdownStructuredSplitter(chunk_size=800, chunk_overlap=100).split(parsed.content)

    assert "# Document title" in parsed.content
    assert "# Main section" in parsed.content
    assert "## Nested section" in parsed.content
    assert "| Field | Value |" in parsed.content
    assert "| keyword | DOCX_PARSER_TEST |" in parsed.content
    assert any(piece.metadata["chunk_type"] == "table" for piece in pieces)
    assert isinstance(parsed.metadata["warnings"], list)


@pytest.mark.asyncio
async def test_pdf_parser_preserves_headings_and_tables(tmp_path: Path) -> None:
    fitz = pytest.importorskip("pymupdf")
    source = tmp_path / "fixture.pdf"
    _write_pdf_fixture(source, fitz)

    parsed = await PdfParser().parse(source.read_bytes(), filename=source.name)
    pieces = MarkdownStructuredSplitter(chunk_size=800, chunk_overlap=100).split(parsed.content)

    assert "# PDF title" in parsed.content
    assert "## Main section" in parsed.content
    assert "PDF_PARSER_TEST" in parsed.content
    assert "| Field | Value |" in parsed.content
    assert "| keyword | PDF_PARSER_TEST |" in parsed.content
    assert any(piece.metadata["chunk_type"] == "table" for piece in pieces)
    assert parsed.metadata["table_count"] == 1


def _write_docx_fixture(path: Path) -> None:
    document_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:pPr><w:pStyle w:val="Title"/></w:pPr><w:r><w:t>Document title</w:t></w:r></w:p>
    <w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Main section</w:t></w:r></w:p>
    <w:p><w:r><w:t>DOCX_PARSER_TEST is searchable.</w:t></w:r></w:p>
    <w:p><w:pPr><w:pStyle w:val="Heading2"/></w:pPr><w:r><w:t>Nested section</w:t></w:r></w:p>
    <w:tbl>
      <w:tr><w:tc><w:p><w:r><w:t>Field</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>Value</w:t></w:r></w:p></w:tc></w:tr>
      <w:tr><w:tc><w:p><w:r><w:t>keyword</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>DOCX_PARSER_TEST</w:t></w:r></w:p></w:tc></w:tr>
    </w:tbl>
  </w:body>
</w:document>"""
    content_types = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml"
    ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>"""
    relationships = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1"
    Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"
    Target="word/document.xml"/>
</Relationships>"""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", relationships)
        archive.writestr("word/document.xml", document_xml)


def _write_pdf_fixture(path: Path, fitz) -> None:
    document = fitz.open()
    try:
        page = document.new_page(width=595, height=842)
        page.insert_text((60, 80), "PDF title", fontsize=23, fontname="hebo")
        page.insert_text((60, 125), "Introductory PDF_PARSER_TEST text.", fontsize=10.5)
        page.insert_text((60, 165), "Main section", fontsize=15, fontname="hebo")
        page.insert_text((60, 205), "The section body remains searchable.", fontsize=10.5)

        x0, y0, x1, y1 = 72, 250, 420, 340
        column_x = 210
        row_y = [y0, y0 + 30, y0 + 60, y1]
        page.draw_rect((x0, y0, x1, y1), color=(0, 0, 0), width=0.8)
        page.draw_line((column_x, y0), (column_x, y1), color=(0, 0, 0), width=0.8)
        for y in row_y[1:-1]:
            page.draw_line((x0, y), (x1, y), color=(0, 0, 0), width=0.8)
        rows = [("Field", "Value"), ("keyword", "PDF_PARSER_TEST")]
        for index, (left, right) in enumerate(rows):
            baseline = y0 + 20 + index * 30
            page.insert_text((x0 + 8, baseline), left, fontsize=9)
            page.insert_text((column_x + 8, baseline), right, fontsize=9)
        document.save(path)
    finally:
        document.close()
