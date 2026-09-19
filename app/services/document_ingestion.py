from __future__ import annotations

import asyncio
from pathlib import Path

from app.core.error_codes import ErrorCode
from app.core.exceptions import AppError, raise_app_error
from app.providers.parsers import ParsedDocument, get_document_parser, source_type_for_filename


async def prepare_document_content(
    *,
    content: str | None,
    file_path: str | None,
    filename: str | None,
    reparse: bool = False,
    mime_type: str | None = None,
) -> ParsedDocument:
    """Resolve stored source bytes or existing text into one parser result."""

    normalized_filename = filename or (Path(file_path).name if file_path else "document.txt")
    should_parse_file = bool(file_path and (reparse or not (content and content.strip())))

    if should_parse_file:
        source_path = Path(file_path)
        try:
            file_bytes = await asyncio.to_thread(source_path.read_bytes)
        except OSError as exc:
            _raise_parse_failed(
                f"cannot read source file {source_path}: {exc}",
                filename=normalized_filename,
            )

        parser = get_document_parser(normalized_filename, mime_type)
        try:
            parsed = await parser.parse(file_bytes, filename=normalized_filename)
        except AppError:
            raise
        except Exception as exc:
            _raise_parse_failed(
                f"failed to parse {normalized_filename}: {exc}",
                filename=normalized_filename,
            )
        if not parsed.content.strip():
            if parsed.source_type == "pdf":
                _raise_parse_failed(
                    "疑似扫描件，本期不支持 OCR",
                    filename=normalized_filename,
                )
            _raise_parse_failed(
                f"parser returned empty content for {normalized_filename}",
                filename=normalized_filename,
            )
        return parsed

    if content and content.strip():
        return ParsedDocument(
            content=content,
            source_type=source_type_for_filename(normalized_filename) or "text",
            metadata={"parser": "stored_content"},
        )

    _raise_parse_failed("document content is empty", filename=normalized_filename)


def _raise_parse_failed(message: str, *, filename: str) -> None:
    raise_app_error(
        ErrorCode.DOCUMENT_PARSE_FAILED,
        "document parsing failed",
        internal_message=message,
        context={"filename": filename},
    )
