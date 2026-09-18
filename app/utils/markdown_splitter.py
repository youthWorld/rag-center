from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.utils.text_splitter import TextSplitter


@dataclass
class SplitPiece:
    """A text piece and the structural metadata discovered while splitting it."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _TextUnit:
    text: str
    atomic: bool = False


class MarkdownStructuredSplitter:
    """Split Markdown by headings and tables before applying size limits."""

    _HEADING_PATTERN = re.compile(r"^\s{0,3}(#{1,6})[ \t]+(.+?)\s*$")
    _FENCE_PATTERN = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
    _TABLE_SEPARATOR_CELL = re.compile(r"^:?-{3,}:?$")

    def __init__(
        self,
        *,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        table_max_rows_per_chunk: int = 10,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be between zero and chunk_size - 1")
        if table_max_rows_per_chunk <= 0:
            raise ValueError("table_max_rows_per_chunk must be greater than zero")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.table_max_rows_per_chunk = table_max_rows_per_chunk

    def split(self, text: str) -> list[SplitPiece]:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        if not normalized.strip():
            return []

        lines = normalized.split("\n")
        pieces: list[SplitPiece] = []
        heading_stack: dict[int, str] = {}
        heading_path: tuple[str, ...] = ()
        section_lines: list[str] = []
        in_fence = False

        for line in lines:
            heading = None if in_fence else self._parse_heading(line)
            if heading is not None and heading[0] <= 2:
                pieces.extend(self._flush_section(section_lines, heading_path))
                level, title = heading
                if level == 1:
                    heading_stack = {1: title}
                else:
                    heading_stack = {
                        stack_level: stack_title
                        for stack_level, stack_title in heading_stack.items()
                        if stack_level < 2
                    }
                    heading_stack[2] = title
                heading_path = tuple(
                    heading_stack[stack_level] for stack_level in sorted(heading_stack)
                )
                section_lines = [line.rstrip()]
                continue

            if not in_fence and line.strip() == "---":
                continue

            section_lines.append(line.rstrip())
            if self._is_fence_line(line):
                in_fence = not in_fence

        pieces.extend(self._flush_section(section_lines, heading_path))
        return pieces

    def _flush_section(
        self,
        lines: list[str],
        heading_path: tuple[str, ...],
    ) -> list[SplitPiece]:
        if not any(line.strip() for line in lines):
            return []

        blocks: list[tuple[str, list[str]]] = []
        text_lines: list[str] = []
        in_fence = False
        index = 0

        while index < len(lines):
            line = lines[index]
            if not in_fence and self._is_table_start(lines, index):
                if text_lines:
                    blocks.append(("text", text_lines))
                    text_lines = []
                table_lines, next_index = self._consume_table(lines, index)
                blocks.append(("table", table_lines))
                index = next_index
                continue

            text_lines.append(line)
            if self._is_fence_line(line):
                in_fence = not in_fence
            index += 1

        if text_lines:
            blocks.append(("text", text_lines))

        pieces: list[SplitPiece] = []
        for block_type, block_lines in blocks:
            if block_type == "table":
                pieces.extend(self._split_table(block_lines, heading_path))
            else:
                pieces.extend(self._split_text_block(block_lines, heading_path))
        return pieces

    def _split_text_block(
        self,
        lines: list[str],
        heading_path: tuple[str, ...],
    ) -> list[SplitPiece]:
        units = self._build_text_units(lines)
        chunks: list[str] = []
        current = ""

        for unit in units:
            if unit.atomic:
                if current:
                    candidate = self._join_units(current, unit.text)
                    if len(candidate) <= self.chunk_size:
                        current = candidate
                    else:
                        chunks.append(current)
                        current = unit.text
                else:
                    current = unit.text
                if len(current) > self.chunk_size:
                    chunks.append(current)
                    current = ""
                continue

            if current and self._is_heading_only(current):
                candidate = self._join_units(current, unit.text)
                if len(candidate) > self.chunk_size:
                    chunks.extend(self._split_text_with_prefix(current, unit.text))
                    current = ""
                    continue

            if len(unit.text) > self.chunk_size:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.extend(self._split_long_text(unit.text))
                continue

            if not current:
                current = unit.text
                continue

            candidate = self._join_units(current, unit.text)
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                chunks.append(current)
                current = self._with_overlap(chunks[-1], unit.text)

        if current:
            chunks.append(current)

        return [
            SplitPiece(
                text=chunk,
                metadata={
                    "heading_path": self._format_heading_path(heading_path),
                    "chunk_type": "section",
                },
            )
            for chunk in chunks
            if chunk.strip()
        ]

    def _build_text_units(self, lines: list[str]) -> list[_TextUnit]:
        units: list[_TextUnit] = []
        current_lines: list[str] = []
        in_fence = False

        def flush(*, atomic: bool = False) -> None:
            if not current_lines:
                return
            text = "\n".join(current_lines).strip()
            if text:
                units.append(_TextUnit(text=text, atomic=atomic))
            current_lines.clear()

        for line in lines:
            if not in_fence:
                if not line.strip():
                    flush()
                    continue
                if self._is_fence_line(line):
                    flush()
                    current_lines.append(line.rstrip())
                    in_fence = True
                    continue
                current_lines.append(line.rstrip())
                continue

            current_lines.append(line.rstrip())
            if self._is_fence_line(line):
                in_fence = False
                flush(atomic=True)

        flush(atomic=in_fence)
        return units

    def _split_text_with_prefix(self, prefix: str, text: str) -> list[str]:
        available = self.chunk_size - len(prefix) - 2
        if available <= 0:
            return [prefix, *self._split_long_text(text)]

        body_chunks = TextSplitter(
            chunk_size=available,
            chunk_overlap=min(self.chunk_overlap, max(available - 1, 0)),
        ).split_text(text)
        if not body_chunks:
            return [prefix]
        return [self._join_units(prefix, body_chunks[0]), *body_chunks[1:]]

    def _split_long_text(self, text: str) -> list[str]:
        heading = self._parse_heading(text.split("\n", 1)[0])
        if heading is None or heading[0] > 2:
            return TextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            ).split_text(text)

        first_line, separator, remainder = text.partition("\n")
        if not separator or not remainder.strip():
            return [text]

        available = self.chunk_size - len(first_line) - 2
        if available <= 0:
            return [first_line, *TextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            ).split_text(remainder)]

        body_chunks = TextSplitter(
            chunk_size=available,
            chunk_overlap=min(self.chunk_overlap, max(available - 1, 0)),
        ).split_text(remainder)
        if not body_chunks:
            return [first_line]
        return [f"{first_line}\n\n{body_chunks[0]}", *body_chunks[1:]]

    def _split_table(
        self,
        lines: list[str],
        heading_path: tuple[str, ...],
    ) -> list[SplitPiece]:
        rows = [line.strip() for line in lines if line.strip()]
        table_text = "\n".join(rows)
        if len(table_text) <= self.chunk_size:
            return [self._table_piece(table_text, heading_path, table_part=1)]

        separator = rows[1] if len(rows) > 1 and self._is_table_separator(rows[1]) else None
        data_start = 2 if separator is not None else 1
        data_rows = rows[data_start:]
        if not data_rows:
            return [self._table_piece(table_text, heading_path, table_part=1)]

        header = self._format_table_header(rows[0])
        pieces: list[SplitPiece] = []
        for part_index, start in enumerate(
            range(0, len(data_rows), self.table_max_rows_per_chunk),
            start=1,
        ):
            group = data_rows[start : start + self.table_max_rows_per_chunk]
            group_text = "\n".join([header, *group])
            pieces.append(self._table_piece(group_text, heading_path, table_part=part_index))
        return pieces

    def _table_piece(
        self,
        text: str,
        heading_path: tuple[str, ...],
        *,
        table_part: int,
    ) -> SplitPiece:
        return SplitPiece(
            text=text,
            metadata={
                "heading_path": self._format_heading_path(heading_path),
                "chunk_type": "table",
                "table_part": table_part,
            },
        )

    def _is_table_start(self, lines: list[str], index: int) -> bool:
        if not self._is_table_row(lines[index]):
            return False
        row_count = 0
        while index + row_count < len(lines) and self._is_table_row(lines[index + row_count]):
            row_count += 1
        return row_count >= 2

    def _consume_table(self, lines: list[str], index: int) -> tuple[list[str], int]:
        table_lines: list[str] = []
        while index < len(lines) and self._is_table_row(lines[index]):
            table_lines.append(lines[index])
            index += 1
        return table_lines, index

    @staticmethod
    def _is_table_row(line: str) -> bool:
        stripped = line.strip()
        return stripped.startswith("|") and "|" in stripped[1:]

    @classmethod
    def _is_table_separator(cls, line: str) -> bool:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        return bool(cells) and all(cls._TABLE_SEPARATOR_CELL.fullmatch(cell) for cell in cells)

    @classmethod
    def _parse_heading(cls, line: str) -> tuple[int, str] | None:
        match = cls._HEADING_PATTERN.match(line)
        if match is None:
            return None
        title = re.sub(r"[ \t]+#+[ \t]*$", "", match.group(2)).strip()
        if not title:
            return None
        return len(match.group(1)), title

    @classmethod
    def _is_fence_line(cls, line: str) -> bool:
        return cls._FENCE_PATTERN.match(line) is not None

    @staticmethod
    def _format_heading_path(heading_path: tuple[str, ...]) -> str | None:
        return "/".join(heading_path) or None

    @staticmethod
    def _format_table_header(header: str) -> str:
        summary = header.strip()
        if summary.startswith("|"):
            summary = summary[1:]
        if summary.endswith("|"):
            summary = summary[:-1]
        return f"【表头】{summary.strip()}"

    @staticmethod
    def _join_units(first: str, second: str) -> str:
        return f"{first}\n\n{second}"

    @classmethod
    def _is_heading_only(cls, text: str) -> bool:
        return cls._parse_heading(text.strip()) is not None

    def _with_overlap(self, previous: str, next_text: str) -> str:
        available = self.chunk_size - len(next_text) - 2
        if self.chunk_overlap <= 0 or available <= 0:
            return next_text
        overlap = previous[-min(self.chunk_overlap, available) :]
        return self._join_units(overlap, next_text)
