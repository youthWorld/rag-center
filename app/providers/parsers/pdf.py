from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

from app.providers.parsers.base import DocumentParser, ParsedDocument


@dataclass(frozen=True)
class _PdfTextBlock:
    lines: tuple[str, ...]
    text: str
    bbox: tuple[float, float, float, float]
    font_size: float
    bold: bool


@dataclass(frozen=True)
class _PdfPageContent:
    blocks: tuple[_PdfTextBlock, ...]
    tables: tuple[str, ...]
    table_events: tuple[tuple[float, float, str], ...]


class PdfParser(DocumentParser):
    def supports(self, filename: str, mime_type: str | None = None) -> bool:
        del mime_type
        return Path(filename).suffix.lower() == ".pdf"

    async def parse(self, file_bytes: bytes, *, filename: str) -> ParsedDocument:
        def extract() -> ParsedDocument:
            try:
                import pymupdf as fitz
            except ImportError:
                try:
                    import fitz
                except ImportError as fallback_exc:
                    raise RuntimeError("PDF parser dependency is not installed") from fallback_exc

            document = fitz.open(stream=file_bytes, filetype="pdf")
            try:
                pages = [_extract_page_content(page) for page in document]
                blocks = [block for page in pages for block in page.blocks]
                body_size = _body_font_size(blocks)
                heading_levels = _heading_levels(blocks, body_size)
                has_structured_headings = any(
                    _heading_level(block, body_size, heading_levels) is not None
                    for block in blocks
                )

                parts: list[str] = []
                for page_index, page in enumerate(pages):
                    if not has_structured_headings and (page.blocks or page.tables):
                        parts.append(f"## 第 {page_index + 1} 页")
                    events: list[tuple[float, float, int, _PdfTextBlock | str]] = [
                        (block.bbox[1], block.bbox[0], 0, block) for block in page.blocks
                    ]
                    events.extend(
                        (y, x, 1, table) for y, x, table in page.table_events
                    )
                    for _y, _x, event_type, event in sorted(
                        events,
                        key=lambda item: (item[0], item[1], item[2]),
                    ):
                        if event_type == 1:
                            parts.append(str(event))
                            continue
                        if not isinstance(event, _PdfTextBlock):
                            continue
                        block = event
                        level = _heading_level(block, body_size, heading_levels)
                        if level is None:
                            parts.append(_format_pdf_block(block))
                        else:
                            parts.append(f"{'#' * level} {block.text}")

                content = "\n\n".join(part.strip() for part in parts if part.strip())
                return ParsedDocument(
                    content=content,
                    source_type="pdf",
                    metadata={
                        "parser": "pymupdf",
                        "page_count": len(document),
                        "table_count": sum(len(page.tables) for page in pages),
                    },
                )
            finally:
                document.close()

        return await asyncio.to_thread(extract)


def _extract_page_content(page: Any) -> _PdfPageContent:
    tables = _extract_tables(page)
    table_bboxes = [bbox for bbox, _rows, _markdown in tables]
    blocks = tuple(
        block
        for block in _extract_text_blocks(page)
        if not any(_bbox_contains(table_bbox, block.bbox) for table_bbox in table_bboxes)
    )
    table_events = tuple(
        (bbox[1], bbox[0], markdown)
        for bbox, _rows, markdown in tables
        if markdown
    )
    return _PdfPageContent(
        blocks=blocks,
        tables=tuple(markdown for _bbox, _rows, markdown in tables if markdown),
        table_events=table_events,
    )


def _extract_text_blocks(page: Any) -> list[_PdfTextBlock]:
    try:
        data = page.get_text("dict", sort=True)
    except TypeError:
        data = page.get_text("dict")

    blocks: list[_PdfTextBlock] = []
    for raw_block in data.get("blocks", []):
        if raw_block.get("type") != 0:
            continue

        lines: list[str] = []
        sizes: list[float] = []
        bold = False
        for raw_line in raw_block.get("lines", []):
            line_parts: list[str] = []
            for span in raw_line.get("spans", []):
                text = str(span.get("text", ""))
                if text:
                    line_parts.append(text)
                size = span.get("size")
                if isinstance(size, (int, float)):
                    sizes.append(float(size))
                font_name = str(span.get("font", "")).lower()
                flags = int(span.get("flags", 0) or 0)
                bold = bold or bool(flags & 16) or "bold" in font_name
            line = "".join(line_parts).strip()
            if line:
                lines.append(line)

        if not lines:
            continue
        bbox = tuple(float(value) for value in raw_block.get("bbox", (0, 0, 0, 0)))
        text = _join_pdf_lines(lines)
        blocks.append(
            _PdfTextBlock(
                lines=tuple(lines),
                text=text,
                bbox=bbox,
                font_size=max(sizes, default=0.0),
                bold=bold,
            )
        )
    return blocks


def _extract_tables(
    page: Any,
) -> list[tuple[tuple[float, float, float, float], list[list[Any]], str]]:
    finder = getattr(page, "find_tables", None)
    if not callable(finder):
        return []
    try:
        result = finder()
        tables = getattr(result, "tables", result)
    except Exception:
        return []

    extracted: list[tuple[tuple[float, float, float, float], list[list[Any]], str]] = []
    for table in tables or []:
        try:
            bbox = tuple(float(value) for value in table.bbox)
            to_markdown = getattr(table, "to_markdown", None)
            if callable(to_markdown):
                rows = table.extract()
                markdown = _normalize_markdown_table(str(to_markdown()))
            else:
                rows = table.extract()
                markdown = _table_to_markdown(rows)
        except Exception:
            continue
        if markdown:
            extracted.append((bbox, rows, markdown))
    return extracted


def _table_to_markdown(rows: Any) -> str:
    if not isinstance(rows, list):
        return ""
    normalized_rows: list[list[str]] = []
    for raw_row in rows:
        if not isinstance(raw_row, (list, tuple)):
            continue
        row = [_normalize_table_cell(value) for value in raw_row]
        if any(row):
            normalized_rows.append(row)
    if not normalized_rows:
        return ""

    column_count = max(len(row) for row in normalized_rows)
    normalized_rows = [row + [""] * (column_count - len(row)) for row in normalized_rows]
    markdown_rows = [
        _format_table_row(normalized_rows[0]),
        _format_table_row(["---"] * column_count),
    ]
    markdown_rows.extend(_format_table_row(row) for row in normalized_rows[1:])
    return "\n".join(markdown_rows)


def _normalize_markdown_table(markdown: str) -> str:
    rows: list[list[str]] = []
    for line in markdown.splitlines():
        if "|" not in line:
            continue
        cells = _split_markdown_row(line)
        if cells and not all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
            rows.append(cells)
    return _table_to_markdown(rows)


def _split_markdown_row(line: str) -> list[str]:
    value = line.strip()
    if value.startswith("|"):
        value = value[1:]
    if value.endswith("|") and not value.endswith(r"\|"):
        value = value[:-1]

    cells: list[str] = []
    current: list[str] = []
    escaped = False
    for character in value:
        if character == "|" and not escaped:
            cells.append("".join(current).strip())
            current.clear()
            continue
        if character == "\\" and not escaped:
            escaped = True
            continue
        current.append(character)
        escaped = False
    cells.append("".join(current).strip())
    return cells


def _format_table_row(values: list[str]) -> str:
    return "| " + " | ".join(values) + " |"


def _normalize_table_cell(value: Any) -> str:
    if value is None:
        return ""
    text = re.sub(r"[ \t\r\n]+", " ", str(value)).strip()
    return text.replace("|", r"\|")


def _bbox_contains(
    outer: tuple[float, float, float, float],
    inner: tuple[float, float, float, float],
) -> bool:
    inner_width = max(inner[2] - inner[0], 0.0)
    inner_height = max(inner[3] - inner[1], 0.0)
    inner_area = inner_width * inner_height
    if inner_area <= 0:
        return False

    intersection_width = max(0.0, min(outer[2], inner[2]) - max(outer[0], inner[0]))
    intersection_height = max(0.0, min(outer[3], inner[3]) - max(outer[1], inner[1]))
    intersection_area = intersection_width * intersection_height
    center_x = (inner[0] + inner[2]) / 2
    center_y = (inner[1] + inner[3]) / 2
    center_inside = outer[0] <= center_x <= outer[2] and outer[1] <= center_y <= outer[3]
    return center_inside or intersection_area / inner_area >= 0.5


def _body_font_size(blocks: list[_PdfTextBlock]) -> float:
    sizes = [block.font_size for block in blocks if block.font_size > 0]
    return median(sizes) if sizes else 10.0


def _heading_levels(
    blocks: list[_PdfTextBlock],
    body_size: float,
) -> tuple[float, ...]:
    candidates = [
        block.font_size
        for block in blocks
        if _is_heading_candidate(block, body_size)
    ]
    levels: list[float] = []
    for size in sorted(set(candidates), reverse=True):
        if not any(abs(size - level) <= 0.75 for level in levels):
            levels.append(size)
    return tuple(levels[:6])


def _heading_level(
    block: _PdfTextBlock,
    body_size: float,
    heading_levels: tuple[float, ...],
) -> int | None:
    if not _is_heading_candidate(block, body_size):
        return None
    for index, size in enumerate(heading_levels, start=1):
        if abs(block.font_size - size) <= 0.75:
            return index
    return None


def _is_heading_candidate(block: _PdfTextBlock, body_size: float) -> bool:
    if not block.text or len(block.text) > 240 or len(block.lines) > 2:
        return False
    if re.match(r"^[•●▪◦]\s*", block.text):
        return False
    if block.font_size <= 0 or body_size <= 0:
        return False
    relative_size = block.font_size / body_size
    return (block.bold and relative_size >= 1.12) or relative_size >= 1.35


def _format_pdf_block(block: _PdfTextBlock) -> str:
    text = block.text
    if re.match(r"^[•●▪◦]\s*", text):
        return re.sub(r"^[•●▪◦]\s*", "- ", text, count=1)
    return text


def _join_pdf_lines(lines: list[str]) -> str:
    if not lines:
        return ""
    result = lines[0]
    for line in lines[1:]:
        if not result:
            result = line
        elif _needs_space(result[-1], line[0]):
            result += " " + line
        else:
            result += line
    return result.strip()


def _needs_space(previous: str, following: str) -> bool:
    if previous.isspace() or following.isspace():
        return False
    if _is_cjk(previous) or _is_cjk(following):
        return False
    return True


def _is_cjk(character: str) -> bool:
    return "\u4e00" <= character <= "\u9fff"
