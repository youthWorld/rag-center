from __future__ import annotations

from pathlib import Path

from app.providers.parsers.base import DocumentParser, ParsedDocument


class MarkdownParser(DocumentParser):
    _SUPPORTED_EXTENSIONS = {".md", ".txt"}

    def supports(self, filename: str, mime_type: str | None = None) -> bool:
        del mime_type
        return Path(filename).suffix.lower() in self._SUPPORTED_EXTENSIONS

    async def parse(self, file_bytes: bytes, *, filename: str) -> ParsedDocument:
        try:
            content = file_bytes.decode("utf-8-sig")
            content = chr(10).join(content.splitlines()).strip()
        except UnicodeDecodeError as exc:
            raise ValueError(f"{filename} is not valid UTF-8 text") from exc

        source_type = "markdown" if Path(filename).suffix.lower() == ".md" else "text"
        return ParsedDocument(
            content=content,
            source_type=source_type,
            metadata={"parser": "plain_text"},
        )
