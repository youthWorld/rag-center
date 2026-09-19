from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ParsedDocument:
    """The normalized parser output consumed by the chunking pipeline."""

    content: str
    source_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


class DocumentParser(ABC):
    @abstractmethod
    def supports(self, filename: str, mime_type: str | None = None) -> bool:
        """Return whether this parser owns the given filename and MIME type."""

    @abstractmethod
    async def parse(self, file_bytes: bytes, *, filename: str) -> ParsedDocument:
        """Parse source bytes into Markdown-style text."""
