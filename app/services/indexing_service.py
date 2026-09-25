from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.error_codes import ErrorCode
from app.core.exceptions import (
    KnowledgeBaseNotFoundError,
    ServiceConfigurationError,
    raise_app_error,
)
from app.core.logging import get_logger
from app.models.document import Document, DocumentStatus
from app.models.index_version import IndexVersionStatus
from app.providers.embedding.base import EmbeddingProvider
from app.providers.keyword_search.base import KeywordSearchProvider
from app.providers.parsers.base import DocumentParser, ParsedDocument
from app.providers.parsers.registry import source_type_for_filename
from app.providers.vectorstores.base import VectorStore
from app.repositories.document_repository import DocumentRepository
from app.repositories.index_version_repository import IndexVersionRepository
from app.repositories.knowledge_base_repository import KnowledgeBaseRepository
from app.schemas.document import DocumentUploadRequest
from app.services.document_ingestion import prepare_document_content
from app.services.reference_relation_service import ReferenceRelationService
from app.utils.id_generator import generate_id
from app.utils.markdown_splitter import (
    MarkdownStructuredSplitter,
    SplitPiece,
    build_retrieval_text,
    is_low_information_content,
    stable_section_id,
)
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
        self.document_parser = document_parser
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
        await self._ensure_not_building(tenant_id=tenant_id, kb_id=knowledge_base.id)

        document = await self.document_repository.create(
            tenant_id=tenant_id,
            kb_id=knowledge_base.id,
            title=request.title,
            content=request.content,
            source_type=source_type_for_filename(request.title) or "text",
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

    async def create_file_document_record(
        self,
        *,
        tenant_id: str,
        kb_id: str,
        title: str,
        source_type: str,
        source_filename: str,
    ) -> Document:
        """Create a processing record whose content will be parsed by the worker."""

        self._require_persistence()
        knowledge_base = await self.knowledge_base_repository.get_by_id_and_tenant(
            kb_id=kb_id,
            tenant_id=tenant_id,
        )
        if knowledge_base is None:
            raise KnowledgeBaseNotFoundError(kb_id)
        await self._ensure_not_building(tenant_id=tenant_id, kb_id=knowledge_base.id)

        document = await self.document_repository.create(
            tenant_id=tenant_id,
            kb_id=knowledge_base.id,
            title=title,
            content=None,
            source_type=source_type,
            source_filename=source_filename,
        )
        await self.session.commit()
        await self.session.refresh(document)
        self.logger.info(
            "BUSINESS_EVENT | event=document_record_created | document_id=%s | kb_id=%s | "
            "tenant_id=%s | source_type=%s",
            document.id,
            document.kb_id,
            tenant_id,
            source_type,
        )
        return document

    async def index_existing_document(
        self,
        document_id: str,
        *,
        reparse: bool = False,
    ) -> int | None:
        """Index a committed document and persist SUCCESS or FAILED state."""

        self._require_document_persistence()
        try:
            document = await self.document_repository.get_by_id(document_id=document_id)
            if document is None or document.status != int(DocumentStatus.PROCESSING):
                return None

            parsed_document = await prepare_document_content(
                content=document.content,
                file_path=getattr(document, "source_file_path", None),
                filename=getattr(document, "source_filename", None) or document.title,
                reparse=reparse,
            )
            if (
                document.content != parsed_document.content
                or document.source_type != parsed_document.source_type
            ):
                document.content = parsed_document.content
                document.source_type = parsed_document.source_type
                await self.session.commit()

            get_kb = getattr(self.knowledge_base_repository, "get_by_id", None)
            knowledge_base = (
                await get_kb(kb_id=document.kb_id, tenant_id=document.tenant_id)
                if callable(get_kb)
                else None
            )
            chunk_count = await self.index_document(
                document,
                parsed_document=parsed_document,
                index_version=getattr(knowledge_base, "active_index_version", "v1") or "v1",
            )
            document.status = int(DocumentStatus.SUCCESS)
            document.error_message = None
            await self.session.commit()
            relation_stats = await self._refresh_relations_if_needed(
                tenant_id=document.tenant_id,
                kb_id=document.kb_id,
                index_version=getattr(knowledge_base, "active_index_version", "v1") or "v1",
            )
            self.logger.info(
                "BUSINESS_EVENT | event=document_indexed | document_id=%s | chunk_count=%s | "
                "index_version=%s | relation_count=%s",
                document.id,
                chunk_count,
                getattr(knowledge_base, "active_index_version", "v1") or "v1",
                (relation_stats or {}).get("relation_count", 0),
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

    async def purge_document_chunks(
        self, document_id: str, *, index_version: str | None = None
    ) -> None:
        """Remove a document from pgvector first and Elasticsearch second."""

        if index_version is None:
            await self.vector_store.delete_by_document_id(document_id)
        else:
            await self.vector_store.delete_by_document_id(document_id, index_version=index_version)
        if self.keyword_search_provider is not None:
            if index_version is None:
                await self.keyword_search_provider.delete_by_document_id(document_id)
            else:
                await self.keyword_search_provider.delete_by_document_id(
                    document_id, index_version=index_version
                )

    async def index_document(
        self,
        document: Document,
        *,
        parsed_document: ParsedDocument | None = None,
        index_version: str = "v1",
    ) -> int:
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
        parsed = parsed_document or ParsedDocument(
            content=document.content or "",
            source_type=document.source_type or "text",
            metadata={"parser": "stored_content"},
        )
        pieces = self._split_document_content(
            document,
            parsed.content,
            parsed.source_type,
            index_version=index_version,
        )
        if not pieces:
            raise ValueError("document content cannot be empty")

        contents = [
            (
                build_retrieval_text(
                    document_title=document.title,
                    heading_path=piece.metadata.get("heading_path"),
                    content=piece.text,
                )
                if index_version != "v1"
                else piece.text
            )
            for piece in pieces
        ]
        embeddings = await self.embedding_provider.embed_documents(contents)
        if len(embeddings) != len(contents):
            raise ValueError("embedding provider returned an unexpected number of vectors")

        source_type = parsed.source_type or document.source_type or "text"
        chunks = [
            {
                "id": generate_id(),
                "tenant_id": document.tenant_id,
                "kb_id": document.kb_id,
                "document_id": document.id,
                "title": document.title,
                "content": piece.text,
                "index_version": index_version,
                "retrieval_text": retrieval_text if index_version != "v1" else None,
                "section_id": piece.metadata.get("section_id"),
                "parent_section_id": piece.metadata.get("parent_section_id"),
                "order_index": piece.metadata.get("order_index"),
                "metadata": {
                    **parsed.metadata,
                    "chunk_index": index,
                    "source_type": source_type,
                    "heading_path": piece.metadata.get("heading_path"),
                    "heading_level": piece.metadata.get("heading_level"),
                    "chunk_type": piece.metadata.get("chunk_type", "section"),
                    "table_part": piece.metadata.get("table_part"),
                    "section_id": piece.metadata.get("section_id"),
                    "parent_section_id": piece.metadata.get("parent_section_id"),
                    "order_index": piece.metadata.get("order_index"),
                    "reference_texts": piece.metadata.get("reference_texts", []),
                },
                "embedding": embedding,
            }
            for index, (piece, retrieval_text, embedding) in enumerate(
                zip(pieces, contents, embeddings, strict=True)
            )
        ]
        for chunk in chunks:
            chunk["metadata"] = {
                key: value for key, value in chunk["metadata"].items() if value is not None
            }
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
        source_type: str | None = None,
        *,
        index_version: str = "v1",
    ) -> list[SplitPiece]:
        if index_version != "v1":
            if self._is_markdown_document(document, source_type=source_type):
                return self.markdown_splitter.split_contextual(
                    content,
                    document_id=document.id,
                    index_version=index_version,
                )
            section_id = stable_section_id(
                document_id=document.id,
                index_version=index_version,
                heading_path=None,
            )
            pieces = [
                SplitPiece(
                    text=piece,
                    metadata={
                        "chunk_type": "prose",
                        "heading_path": None,
                        "section_id": section_id,
                        "parent_section_id": None,
                        "reference_texts": [],
                    },
                )
                for piece in self.splitter.split_text(content)
                if not is_low_information_content(piece)
            ]
            for order_index, piece in enumerate(pieces):
                piece.metadata["order_index"] = order_index
            return pieces
        if self._is_markdown_document(document, source_type=source_type):
            return self.markdown_splitter.split(content)
        return [
            SplitPiece(
                text=piece,
                metadata={"chunk_type": "section", "heading_path": None},
            )
            for piece in self.splitter.split_text(content)
        ]

    @staticmethod
    def _is_markdown_document(document: Document, *, source_type: str | None = None) -> bool:
        source_type = (source_type or document.source_type or "").lower()
        if source_type in {"markdown", "md", "docx", "pdf"}:
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

    async def _refresh_relations_if_needed(
        self, *, tenant_id: str, kb_id: str, index_version: str
    ) -> dict[str, int] | None:
        if index_version == "v1" or self.session is None:
            return None
        try:
            stats = await ReferenceRelationService(self.session).build_relations(
                tenant_id=tenant_id,
                kb_id=kb_id,
                index_version=index_version,
            )
            await self.session.commit()
            return stats
        except Exception as exc:
            await self.session.rollback()
            self.logger.warning(
                "BUSINESS_EVENT | event=reference_relation_refresh_degraded | "
                "tenant_id=%s | kb_id=%s | index_version=%s | error=%s",
                tenant_id,
                kb_id,
                index_version,
                type(exc).__name__,
            )
            return None

    def _require_persistence(self) -> None:
        self._require_document_persistence()
        if self.knowledge_base_repository is None:
            raise RuntimeError("IndexingService knowledge base dependency is not configured")

    async def _ensure_not_building(self, *, tenant_id: str, kb_id: str) -> None:
        if self.session is None:
            return
        versions = await IndexVersionRepository(self.session).list(tenant_id=tenant_id, kb_id=kb_id)
        if any(item.status == IndexVersionStatus.BUILDING for item in versions):
            raise_app_error(ErrorCode.SYSTEM_BUSY, "index rebuild is in progress")

    def _require_document_persistence(self) -> None:
        if self.session is None or self.document_repository is None:
            raise RuntimeError("IndexingService persistence dependencies are not configured")


def build_indexing_service(session: AsyncSession, app_settings: Settings) -> IndexingService:
    from app.providers.embedding.openai_compatible import OpenAICompatibleEmbeddingProvider
    from app.providers.keyword_search.elasticsearch import ElasticsearchKeywordSearchProvider
    from app.providers.vectorstores.pgvector import PgVectorStore

    if app_settings.keyword_search_provider != "elasticsearch":
        raise ServiceConfigurationError(
            internal_message=(
                f"unsupported KEYWORD_SEARCH_PROVIDER: {app_settings.keyword_search_provider}"
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
