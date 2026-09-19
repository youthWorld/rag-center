from __future__ import annotations

from pathlib import Path

from app.providers.parsers.base import DocumentParser
from app.providers.parsers.docx import DocxParser
from app.providers.parsers.markdown import MarkdownParser
from app.providers.parsers.noop import NoopParser
from app.providers.parsers.pdf import PdfParser

_PARSERS: tuple[DocumentParser, ...] = (DocxParser(), PdfParser(), MarkdownParser())
_SOURCE_TYPES = {
    ".docx": "docx",
    ".pdf": "pdf",
    ".md": "markdown",
    ".txt": "text",
}


def get_document_parser(
    filename: str,
    mime_type: str | None = None,
) -> DocumentParser:
    for parser in _PARSERS:
        if parser.supports(filename, mime_type):
            return parser
    return NoopParser()


def is_supported_document(filename: str, mime_type: str | None = None) -> bool:
    return not isinstance(get_document_parser(filename, mime_type), NoopParser)


def source_type_for_filename(filename: str) -> str | None:
    return _SOURCE_TYPES.get(Path(filename).suffix.lower())


def supported_extensions() -> tuple[str, ...]:
    return tuple(_SOURCE_TYPES)
