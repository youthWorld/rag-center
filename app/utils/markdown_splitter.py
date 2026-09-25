from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

from app.utils.text_splitter import TextSplitter


@dataclass
class SplitPiece:
    """A text piece and the structural metadata discovered while splitting it."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def normalize_content(content: str) -> str:
    """Normalize only for quality checks; never use this value as user content."""

    return re.sub(r"\s+", " ", (content or "").strip())


def strip_markdown(value: str) -> str:
    value = re.sub(r"^\s{0,3}#{1,6}\s+", "", value.strip())
    value = re.sub(r"[*_`~]", "", value)
    return normalize_content(value)


def contains_letter_number_or_cjk(value: str) -> bool:
    return bool(re.search(r"[A-Za-z0-9\u3400-\u9fff]", value or ""))


def is_separator_only(value: str) -> bool:
    lines = [line.strip() for line in (value or "").splitlines() if line.strip()]
    return bool(lines) and all(bool(re.fullmatch(r"[-_*~`\s|]+", line)) for line in lines)


def is_low_information_content(content: str, heading: str | None = None) -> bool:
    """Return whether a piece has no standalone evidence.

    This deliberately does not use a minimum character count: short policy
    statements can be highly informative.
    """

    normalized = normalize_content(content)
    if not normalized or is_separator_only(content):
        return True
    if not contains_letter_number_or_cjk(normalized):
        return True
    if heading and strip_markdown(normalized) == strip_markdown(heading):
        return True
    lines = [line.strip() for line in (content or "").splitlines() if line.strip()]
    if (
        len(lines) == 2
        and lines[0].startswith("|")
        and MarkdownStructuredSplitter._is_table_separator(lines[1])
    ):
        return True
    if lines and len(lines) == 1 and MarkdownStructuredSplitter._parse_heading(lines[0]):
        return True
    return False


def build_retrieval_text(
    *, document_title: str | None, heading_path: str | None, content: str
) -> str:
    """Build the deterministic text sent to vector, BM25 and rerank providers."""

    lines: list[str] = []
    if document_title and document_title.strip():
        lines.append(f"文档：{document_title.strip()}")
    if heading_path and heading_path.strip():
        lines.append(f"章节：{heading_path.replace('/', ' > ')}")
    lines.append(f"正文：{content}")
    return "\n".join(lines)


def stable_section_id(*, document_id: str, index_version: str, heading_path: str | None) -> str:
    path = heading_path or "__root__"
    digest = hashlib.sha1(f"{document_id}:{index_version}:{path}".encode()).hexdigest()[:24]
    return f"section-{digest}"


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

    def split(self, text: str, *, include_subheadings: bool = False) -> list[SplitPiece]:
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
            if heading is not None and (include_subheadings or heading[0] <= 2):
                pieces.extend(self._flush_section(section_lines, heading_path))
                level, title = heading
                if level == 1:
                    heading_stack = {1: title}
                else:
                    heading_stack = {
                        stack_level: stack_title
                        for stack_level, stack_title in heading_stack.items()
                        if stack_level < level
                    }
                    heading_stack[level] = title
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

    def split_contextual(
        self,
        text: str,
        *,
        document_id: str,
        index_version: str = "v2",
    ) -> list[SplitPiece]:
        """Create index-ready contextual pieces without changing raw content.

        ` ` split ` ` remains the backwards-compatible structural parser. This
        method applies the v2 rules: low-information filtering, same-section
        coalescing, deterministic section identifiers and final ordering.
        """

        raw_pieces = self.split(text, include_subheadings=True)
        if not raw_pieces:
            return []

        coalesced: list[SplitPiece] = []
        seen_by_section: dict[str, set[str]] = {}
        pending_reference: dict[str, str] = {}
        for piece in raw_pieces:
            heading_path = piece.metadata.get("heading_path")
            heading = heading_path.rsplit("/", 1)[-1] if heading_path else None
            if is_low_information_content(piece.text, heading):
                continue
            section_id = stable_section_id(
                document_id=document_id,
                index_version=index_version,
                heading_path=heading_path,
            )
            normalized = normalize_content(piece.text)
            section_seen = seen_by_section.setdefault(section_id, set())
            if normalized in section_seen:
                continue
            section_seen.add(normalized)

            is_pointer = self._is_reference_pointer_only(piece.text)
            if is_pointer and coalesced and coalesced[-1].metadata.get("section_id") == section_id:
                coalesced[-1].text = self._join_units(coalesced[-1].text, piece.text)
                coalesced[-1].metadata["reference_texts"] = list(
                    dict.fromkeys(
                        [
                            *coalesced[-1].metadata.get("reference_texts", []),
                            *self._reference_texts(piece.text),
                        ]
                    )
                )
                continue
            if is_pointer:
                pending_reference[section_id] = (
                    self._join_units(pending_reference[section_id], piece.text)
                    if section_id in pending_reference
                    else piece.text
                )
                continue
            if section_id in pending_reference:
                piece.text = self._join_units(piece.text, pending_reference.pop(section_id))

            if coalesced:
                previous = coalesced[-1]
                same_section = previous.metadata.get("section_id") == section_id
                candidate = self._join_units(previous.text, piece.text)
                if same_section and len(candidate) <= self.chunk_size:
                    previous.text = candidate
                    previous.metadata["chunk_type"] = self._merged_chunk_type(
                        previous.metadata.get("chunk_type"),
                        piece.metadata.get("chunk_type"),
                    )
                    previous.metadata["reference_texts"] = [
                        *previous.metadata.get("reference_texts", []),
                        *self._reference_texts(piece.text),
                    ]
                    continue

            parent_path = (
                heading_path.rsplit("/", 1)[0]
                if isinstance(heading_path, str) and "/" in heading_path
                else None
            )
            metadata = {
                **piece.metadata,
                "section_id": section_id,
                "parent_section_id": (
                    stable_section_id(
                        document_id=document_id,
                        index_version=index_version,
                        heading_path=parent_path,
                    )
                    if parent_path
                    else None
                ),
                "reference_texts": self._reference_texts(piece.text),
                "chunk_type": self._contextual_chunk_type(piece.text, piece.metadata),
                "heading_level": len(heading_path.split("/")) if heading_path else None,
            }
            coalesced.append(SplitPiece(text=piece.text, metadata=metadata))

        for order_index, piece in enumerate(coalesced):
            piece.metadata["order_index"] = order_index
        return coalesced

    @staticmethod
    def _reference_texts(content: str) -> list[str]:
        matches = re.findall(
            r"(?:^|[>\n])\s*((?:关联规则|参见|见)[：:：]?[^\n]+|\[[^]]+\]\([^)]*\))",
            content,
        )
        return [normalize_content(match) for match in matches if normalize_content(match)]

    @classmethod
    def _is_reference_pointer_only(cls, content: str) -> bool:
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        evidence_lines = [line for line in lines if cls._parse_heading(line) is None]
        return bool(evidence_lines) and all(
            re.match(r"^\s*>?\s*(?:参见|见|关联规则)[：:]", line)
            for line in evidence_lines
        )

    @staticmethod
    def _contextual_chunk_type(content: str, metadata: dict[str, Any]) -> str:
        if metadata.get("chunk_type") == "table" or ("|" in content and "【表头】" in content):
            return "table"
        if re.search(r"^\s*```", content, flags=re.MULTILINE):
            return "code"
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if lines and all(re.match(r"^(?:[-*+] |\d+[.)] )", line) for line in lines):
            return "list"
        return "prose"

    @staticmethod
    def _merged_chunk_type(first: str | None, second: str | None) -> str:
        if first == second and first:
            return first
        return "prose"

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
            return [
                first_line,
                *TextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                ).split_text(remainder),
            ]

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
