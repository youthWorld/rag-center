from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.exceptions import KnowledgeBaseNotFoundError, ServiceConfigurationError
from app.core.logging import get_logger
from app.models.document import Document, DocumentStatus
from app.providers.embedding.base import EmbeddingProvider
from app.providers.keyword_search.base import KeywordSearchProvider
from app.providers.parsers.base import DocumentParser
from app.providers.parsers.plain_text import PlainTextDocumentParser
from app.providers.vectorstores.base import VectorStore
from app.repositories.document_repository import DocumentRepository
from app.repositories.knowledge_base_repository import KnowledgeBaseRepository
from app.schemas.document import DocumentUploadRequest
from app.utils.id_generator import generate_id
from app.utils.markdown_splitter import MarkdownStructuredSplitter, SplitPiece
from app.utils.text_splitter import TextSplitter


class IndexingService:
    """Coordinate document records, chunking, embeddings, and both indexes."""

    def __init__(
        self,
        *,
        splitter: TextSplitter,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        markdown_splitter: MarkdownStructuredSplitter | None = None,
        keyword_search_provider: KeywordSearchProvider | None = None,
        document_parser: DocumentParser | None = None,
        session: AsyncSession | None = None,
        document_repository: DocumentRepository | None = None,
        knowledge_base_repository: KnowledgeBaseRepository | None = None,
    ) -> None:
        self.session = session
        self.splitter = splitter
        self.markdown_splitter = markdown_splitter or MarkdownStructuredSplitter(
            chunk_size=splitter.chunk_size,
            chunk_overlap=splitter.chunk_overlap,
        )
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.keyword_search_provider = keyword_search_provider
        self.document_parser = document_parser or PlainTextDocumentParser()
        self.document_repository = document_repository or (
            DocumentRepository(session) if session is not None else None
        )
        self.knowledge_base_repository = knowledge_base_repository or (
            KnowledgeBaseRepository(session) if session is not None else None
        )
        self.logger = get_logger(__name__)

    async def create_document_record(
        self,
        tenant_id: str,
        request: DocumentUploadRequest,
    ) -> Document:
        """Validate ownership and commit a PROCESSING document before dispatch."""

        self._require_persistence()
        knowledge_base = await self.knowledge_base_repository.get_by_id_and_tenant(
            kb_id=request.kb_id,
            tenant_id=tenant_id,
        )
        if knowledge_base is None:
            raise KnowledgeBaseNotFoundError(request.kb_id)

        document = await self.document_repository.create(
            tenant_id=tenant_id,
            kb_id=knowledge_base.id,
            title=request.title,
            content=request.content,
        )
        await self.session.commit()
        await self.session.refresh(document)
        self.logger.info(
            "BUSINESS_EVENT | event=document_record_created | document_id=%s | kb_id=%s | "
            "tenant_id=%s",
            document.id,
            document.kb_id,
            tenant_id,
        )
        return document

    async def index_existing_document(self, document_id: str) -> int | None:
        """Index a committed document and persist SUCCESS or FAILED state."""

        self._require_document_persistence()
        try:
            document = await self.document_repository.get_by_id(document_id=document_id)
            if document is None or document.status != int(DocumentStatus.PROCESSING):
                return None

            chunk_count = await self.index_document(document)
            document.status = int(DocumentStatus.SUCCESS)
            document.error_message = None
            await self.session.commit()
            self.logger.info(
                "BUSINESS_EVENT | event=document_indexed | document_id=%s | chunk_count=%s",
                document.id,
                chunk_count,
            )
            return chunk_count
        except Exception as exc:
            await self._mark_failed(document_id, exc)
            self.logger.exception(
                "BUSINESS_ERROR | event=document_indexing_failed | document_id=%s",
                document_id,
            )
            raise

    async def mark_document_failed(self, document_id: str, error_message: str) -> None:
        self._require_document_persistence()
        await self._mark_failed(document_id, RuntimeError(error_message))

    async def mark_document_processing(self, document_id: str) -> Document | None:
        self._require_document_persistence()
        document = await self.document_repository.get_by_id(document_id=document_id)
        if document is None:
            return None
        document.status = int(DocumentStatus.PROCESSING)
        document.error_message = None
        await self.session.commit()
        return document

    async def purge_document_chunks(self, document_id: str) -> None:
        """Remove a document from pgvector first and Elasticsearch second."""

        await self.vector_store.delete_by_document_id(document_id)
        if self.keyword_search_provider is not None:
            await self.keyword_search_provider.delete_by_document_id(document_id)

    async def index_document(self, document: Document) -> int:
        """Chunk, embed, and persist one document's content.

        This method remains as the low-level compatibility entry point used by
        existing provider tests. Lifecycle callers should use the two methods
        above so record creation and indexing can run in different processes.
        """

        self.logger.info(
            "BUSINESS_EVENT | event=document_indexing_started | document_id=%s | kb_id=%s",
            document.id,
            document.kb_id,
        )
        parsed_content = self.document_parser.parse(
            document.content,
            source_type=document.source_type or "text",
        )
        pieces = self._split_document_content(document, parsed_content)
        if not pieces:
            raise ValueError("document content cannot be empty")

        contents = [piece.text for piece in pieces]
        embeddings = await self.embedding_provider.embed_documents(contents)
        if len(embeddings) != len(contents):
            raise ValueError("embedding provider returned an unexpected number of vectors")

        source_type = document.source_type or "text"
        chunks = [
            {
                "id": generate_id(),
                "tenant_id": document.tenant_id,
                "kb_id": document.kb_id,
                "document_id": document.id,
                "title": document.title,
                "content": piece.text,
                "metadata": {
                    "chunk_index": index,
                    "source_type": source_type,
                    "heading_path": piece.metadata.get("heading_path"),
                    "chunk_type": piece.metadata.get("chunk_type", "section"),
                    "table_part": piece.metadata.get("table_part"),
                },
                "embedding": embedding,
            }
            for index, (piece, embedding) in enumerate(
                zip(pieces, embeddings, strict=True)
            )
        ]
        await self.vector_store.add_chunks(chunks)
        if self.keyword_search_provider is not None:
            await self.keyword_search_provider.add_chunks(chunks)
        self.logger.info(
            "BUSINESS_EVENT | event=document_chunks_persisted | document_id=%s | chunk_count=%s",
            document.id,
            len(chunks),
        )
        return len(chunks)

    def _split_document_content(
        self,
        document: Document,
        content: str,
    ) -> list[SplitPiece]:
        if self._is_markdown_document(document):
            return self.markdown_splitter.split(content)
        return [
            SplitPiece(
                text=piece,
                metadata={"chunk_type": "section", "heading_path": None},
            )
            for piece in self.splitter.split_text(content)
        ]

    @staticmethod
    def _is_markdown_document(document: Document) -> bool:
        source_type = (document.source_type or "").lower()
        if source_type in {"markdown", "md"}:
            return True
        title = (document.title or "").lower()
        return title.endswith((".md", ".markdown"))

    async def _mark_failed(self, document_id: str, exception: Exception) -> None:
        await self.session.rollback()
        document = await self.document_repository.get_by_id(document_id=document_id)
        if document is None:
            return
        document.status = int(DocumentStatus.FAILED)
        document.error_message = str(exception)[:2000]
        await self.session.commit()

    def _require_persistence(self) -> None:
        self._require_document_persistence()
        if self.knowledge_base_repository is None:
            raise RuntimeError("IndexingService knowledge base dependency is not configured")

    def _require_document_persistence(self) -> None:
        if (
            self.session is None
            or self.document_repository is None
        ):
            raise RuntimeError("IndexingService persistence dependencies are not configured")


def build_indexing_service(session: AsyncSession, app_settings: Settings) -> IndexingService:
    from app.providers.embedding.openai_compatible import OpenAICompatibleEmbeddingProvider
    from app.providers.keyword_search.elasticsearch import ElasticsearchKeywordSearchProvider
    from app.providers.vectorstores.pgvector import PgVectorStore

    if app_settings.keyword_search_provider != "elasticsearch":
        raise ServiceConfigurationError(
            internal_message=(
                "unsupported KEYWORD_SEARCH_PROVIDER: "
                f"{app_settings.keyword_search_provider}"
            ),
            context={"provider": app_settings.keyword_search_provider},
        )

    return IndexingService(
        session=session,
        document_repository=DocumentRepository(session),
        knowledge_base_repository=KnowledgeBaseRepository(session),
        splitter=TextSplitter(
            chunk_size=app_settings.chunk_size,
            chunk_overlap=app_settings.chunk_overlap,
        ),
        markdown_splitter=MarkdownStructuredSplitter(
            chunk_size=app_settings.chunk_size,
            chunk_overlap=app_settings.chunk_overlap,
            table_max_rows_per_chunk=app_settings.table_max_rows_per_chunk,
        ),
        embedding_provider=OpenAICompatibleEmbeddingProvider(app_settings),
        vector_store=PgVectorStore(session),
        keyword_search_provider=ElasticsearchKeywordSearchProvider(app_settings),
    )
