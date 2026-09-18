from app.providers.parsers.base import DocumentParser


class PlainTextDocumentParser(DocumentParser):
    def parse(self, content: str, *, source_type: str) -> str:
        if source_type not in {"text", "markdown", "md"}:
            raise ValueError(f"unsupported document source type: {source_type}")
        return content
