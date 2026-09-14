from abc import ABC, abstractmethod


class DocumentParser(ABC):
    @abstractmethod
    def parse(self, content: str, *, source_type: str) -> str:
        """Convert a source document into text ready for chunking."""
