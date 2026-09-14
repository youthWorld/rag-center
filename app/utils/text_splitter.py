class TextSplitter:
    """Split text into overlapping character windows while respecting natural breaks."""

    def __init__(self, *, chunk_size: int = 800, chunk_overlap: int = 100) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be between zero and chunk_size - 1")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> list[str]:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not normalized:
            return []
        if len(normalized) <= self.chunk_size:
            return [normalized]

        chunks: list[str] = []
        start = 0
        text_length = len(normalized)

        while start < text_length:
            target_end = min(start + self.chunk_size, text_length)
            end = target_end

            if target_end < text_length:
                lower_bound = start + max(self.chunk_size // 2, 1)
                boundaries = (
                    normalized.rfind("\n\n", start + 1, target_end + 1),
                    normalized.rfind("\n", start + 1, target_end + 1),
                    normalized.rfind(" ", start + 1, target_end + 1),
                )
                candidates = [boundary for boundary in boundaries if boundary >= lower_bound]
                if candidates:
                    end = max(candidates)

            piece = normalized[start:end].strip()
            if piece and (not chunks or piece != chunks[-1]):
                chunks.append(piece)

            if end >= text_length:
                break
            start = max(end - self.chunk_overlap, start + 1)

        return chunks
