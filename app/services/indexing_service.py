from app.core.logging import get_logger
from app.models.document import Document
from app.providers.embedding.base import EmbeddingProvider
from app.providers.parsers.base import DocumentParser
from app.providers.parsers.plain_text import PlainTextDocumentParser
from app.providers.vectorstores.base import VectorStore
from app.utils.id_generator import generate_id
from app.utils.text_splitter import TextSplitter


class IndexingService:
    """Synchronously orchestrate chunking, embeddings, and vector persistence."""

    def __init__(
        self,
        *,
        splitter: TextSplitter,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        document_parser: DocumentParser | None = None,
    ) -> None:
        self.splitter = splitter
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.document_parser = document_parser or PlainTextDocumentParser()
        self.logger = get_logger(__name__)

    async def index_document(self, document: Document) -> int:
        self.logger.info(
            "BUSINESS_EVENT | event=document_indexing_started | document_id=%s | kb_id=%s",
            document.id,
            document.kb_id,
        )
        parsed_content = self.document_parser.parse(
            document.content,
            source_type=document.source_type or "text",
        )
        contents = self.splitter.split_text(parsed_content)
        if not contents:
            raise ValueError("document content cannot be empty")

        embeddings = await self.embedding_provider.embed_documents(contents)
        if len(embeddings) != len(contents):
            raise ValueError("embedding provider returned an unexpected number of vectors")

        chunks = [
            {
                "id": generate_id(),
                "tenant_id": document.tenant_id,
                "kb_id": document.kb_id,
                "document_id": document.id,
                "title": document.title,
                "content": content,
                "metadata": {"chunk_index": index},
                "embedding": embedding,
            }
            for index, (content, embedding) in enumerate(zip(contents, embeddings, strict=True))
        ]
        await self.vector_store.add_chunks(chunks)
        self.logger.info(
            "BUSINESS_EVENT | event=document_chunks_persisted | document_id=%s | chunk_count=%s",
            document.id,
            len(chunks),
        )
        return len(chunks)
