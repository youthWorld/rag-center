from __future__ import annotations

from app.core.error_codes import ErrorCode
from app.core.exceptions import raise_app_error
from app.providers.parsers.base import DocumentParser, ParsedDocument


class NoopParser(DocumentParser):
    def supports(self, filename: str, mime_type: str | None = None) -> bool:
        del filename, mime_type
        return False

    async def parse(self, file_bytes: bytes, *, filename: str) -> ParsedDocument:
        del file_bytes
        raise_app_error(
            ErrorCode.PARAM_ERROR,
            f"unsupported document format: {filename}",
            context={"filename": filename},
        )
