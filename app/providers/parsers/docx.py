from __future__ import annotations

import asyncio
import io
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path

from app.providers.parsers.base import DocumentParser, ParsedDocument

_DOCX_STYLE_MAP = "\n".join(
    (
        "p.Title => h1:fresh",
        "p[style-name='Title'] => h1:fresh",
        "p[style-name='标题'] => h1:fresh",
        "p[style-name='标题 1'] => h1:fresh",
        "p[style-name='标题1'] => h1:fresh",
        "p[style-name='标题 2'] => h2:fresh",
        "p[style-name='标题2'] => h2:fresh",
        "p[style-name='标题 3'] => h3:fresh",
        "p[style-name='标题3'] => h3:fresh",
    )
)


@dataclass
class _HtmlNode:
    tag: str
    attributes: dict[str, str] = field(default_factory=dict)
    children: list[_HtmlNode | str] = field(default_factory=list)


class _HtmlTreeParser(HTMLParser):
    _VOID_TAGS = {"br", "hr", "img", "input", "link", "meta"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = _HtmlNode("root")
        self._stack = [self.root]

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        node = _HtmlNode(tag.lower(), {key: value or "" for key, value in attrs})
        self._stack[-1].children.append(node)
        if node.tag not in self._VOID_TAGS:
            self._stack.append(node)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if self._stack[-1].tag == tag.lower():
            self._stack.pop()

    def handle_endtag(self, tag: str) -> None:
        normalized_tag = tag.lower()
        for index in range(len(self._stack) - 1, 0, -1):
            if self._stack[index].tag == normalized_tag:
                del self._stack[index:]
                return

    def handle_data(self, data: str) -> None:
        self._stack[-1].children.append(data)


class DocxParser(DocumentParser):
    def supports(self, filename: str, mime_type: str | None = None) -> bool:
        del mime_type
        return Path(filename).suffix.lower() == ".docx"

    async def parse(self, file_bytes: bytes, *, filename: str) -> ParsedDocument:
        def convert() -> ParsedDocument:
            try:
                import mammoth
            except ImportError as exc:
                raise RuntimeError(
                    "DOCX parser dependency is not installed; install the 'parsers' extra"
                ) from exc

            result = mammoth.convert_to_html(
                io.BytesIO(file_bytes),
                style_map=_DOCX_STYLE_MAP,
            )
            warnings = [getattr(message, "message", str(message)) for message in result.messages]
            content = _html_to_markdown(result.value)
            return ParsedDocument(
                content=content,
                source_type="docx",
                metadata={"parser": "mammoth", "warnings": warnings},
            )

        return await asyncio.to_thread(convert)


def _html_to_markdown(markup: str) -> str:
    parser = _HtmlTreeParser()
    parser.feed(markup)
    parser.close()
    return _render_document(parser.root).strip()


def _render_document(node: _HtmlNode) -> str:
    blocks: list[str] = []
    inline_buffer: list[str] = []

    def flush_inline() -> None:
        text = _normalize_inline("".join(inline_buffer))
        inline_buffer.clear()
        if text:
            blocks.append(text)

    for child in node.children:
        if isinstance(child, str):
            inline_buffer.append(child)
            continue
        if child.tag in _BLOCK_TAGS:
            flush_inline()
            rendered = _render_block(child).strip()
            if rendered:
                blocks.append(rendered)
        else:
            inline_buffer.append(_render_inline(child))

    flush_inline()
    return "\n\n".join(blocks)


_BLOCK_TAGS = {
    "blockquote",
    "div",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "li",
    "ol",
    "p",
    "pre",
    "table",
    "ul",
}


def _render_block(node: _HtmlNode) -> str:
    if node.tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
        level = int(node.tag[1])
        text = _normalize_inline(_render_inline_children(node))
        return f"{'#' * level} {text}" if text else ""
    if node.tag == "p":
        return _normalize_inline(_render_inline_children(node))
    if node.tag == "table":
        return _render_table(node)
    if node.tag in {"ul", "ol"}:
        return _render_list(node)
    if node.tag == "li":
        return _normalize_inline(_render_inline_children(node))
    if node.tag == "blockquote":
        content = _render_document(node)
        return "\n".join(f"> {line}" for line in content.splitlines())
    if node.tag == "pre":
        content = _raw_text(node).strip("\n")
        return f"```\n{content}\n```" if content else ""
    return _render_document(node)


def _render_inline(node: _HtmlNode) -> str:
    if node.tag == "br":
        return "\n"
    if node.tag in {"strong", "b"}:
        text = _normalize_inline(_render_inline_children(node))
        return f"**{text}**" if text else ""
    if node.tag in {"em", "i"}:
        text = _normalize_inline(_render_inline_children(node))
        return f"*{text}*" if text else ""
    if node.tag == "code":
        return f"`{_raw_text(node).strip()}`"
    if node.tag == "a":
        text = _normalize_inline(_render_inline_children(node))
        href = node.attributes.get("href", "").strip()
        return f"[{text}]({href})" if text and href else text
    if node.tag == "img":
        alt = node.attributes.get("alt", "").strip()
        src = node.attributes.get("src", "").strip()
        return f"![{alt}]({src})" if alt or src else ""
    if node.tag in _BLOCK_TAGS:
        return _render_block(node)
    return _render_inline_children(node)


def _render_inline_children(node: _HtmlNode) -> str:
    return "".join(
        child if isinstance(child, str) else _render_inline(child)
        for child in node.children
    )


def _render_list(node: _HtmlNode) -> str:
    ordered = node.tag == "ol"
    lines: list[str] = []
    item_index = 1
    for child in node.children:
        if not isinstance(child, _HtmlNode) or child.tag != "li":
            continue

        item_parts: list[str] = []
        nested_lists: list[str] = []
        for item_child in child.children:
            if isinstance(item_child, _HtmlNode) and item_child.tag in {"ul", "ol"}:
                nested_lists.append(_render_list(item_child))
            else:
                item_parts.append(
                    item_child
                    if isinstance(item_child, str)
                    else _render_inline(item_child)
                )

        text = _normalize_inline("".join(item_parts))
        if text:
            prefix = f"{item_index}. " if ordered else "- "
            lines.append(prefix + text)
            item_index += 1
        for nested in nested_lists:
            lines.extend(f"  {line}" for line in nested.splitlines() if line.strip())
    return "\n".join(lines)


def _render_table(node: _HtmlNode) -> str:
    rows: list[list[str]] = []
    for row in _find_nodes(node, "tr"):
        cells = [
            child
            for child in row.children
            if isinstance(child, _HtmlNode) and child.tag in {"td", "th"}
        ]
        if not cells:
            continue
        rows.append([_normalize_table_cell(_render_inline_children(cell)) for cell in cells])

    if not rows:
        return ""

    column_count = max(len(row) for row in rows)
    normalized_rows = [row + [""] * (column_count - len(row)) for row in rows]
    header = normalized_rows[0]
    separator = ["---"] * column_count
    markdown_rows = [_format_table_row(header), _format_table_row(separator)]
    markdown_rows.extend(_format_table_row(row) for row in normalized_rows[1:])
    return "\n".join(markdown_rows)


def _find_nodes(node: _HtmlNode, tag: str) -> list[_HtmlNode]:
    nodes: list[_HtmlNode] = []
    for child in node.children:
        if isinstance(child, _HtmlNode):
            if child.tag == tag:
                nodes.append(child)
            nodes.extend(_find_nodes(child, tag))
    return nodes


def _format_table_row(values: list[str]) -> str:
    return "| " + " | ".join(values) + " |"


def _normalize_table_cell(value: str) -> str:
    value = _normalize_inline(value).replace("|", r"\|")
    return value.replace("\n", "<br>").strip()


def _normalize_inline(value: str) -> str:
    return re.sub(r"[ \t\r\n]+", " ", value).strip()


def _raw_text(node: _HtmlNode) -> str:
    return "".join(
        child if isinstance(child, str) else _raw_text(child)
        for child in node.children
    )
