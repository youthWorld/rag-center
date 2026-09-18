from app.models.document import Document
from app.providers.embedding.base import EmbeddingProvider
from app.providers.keyword_search.base import KeywordSearchProvider
from app.providers.vectorstores.base import VectorStore
from app.services.indexing_service import IndexingService
from app.utils.markdown_splitter import MarkdownStructuredSplitter
from app.utils.text_splitter import TextSplitter


class FakeEmbeddingProvider(EmbeddingProvider):
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(index)] for index, _ in enumerate(texts)]

    async def embed_query(self, query: str) -> list[float]:
        return [float(len(query))]


class FakeVectorStore(VectorStore):
    def __init__(self) -> None:
        self.chunks: list[dict] = []

    async def add_chunks(self, chunks: list[dict]) -> None:
        self.chunks.extend(chunks)

    async def similarity_search(self, query_vector, *, tenant_id, kb_id, top_k=5):
        return self.chunks[:top_k]

    async def delete_by_document_id(self, document_id: str) -> None:
        self.chunks = [chunk for chunk in self.chunks if chunk["document_id"] != document_id]


class FakeKeywordSearchProvider(KeywordSearchProvider):
    def __init__(self) -> None:
        self.chunks: list[dict] = []

    async def add_chunks(self, chunks: list[dict]) -> None:
        self.chunks.extend(chunks)

    async def keyword_search(self, *, query, tenant_id, kb_id, top_k=20):
        del query, tenant_id, kb_id, top_k
        return []

    async def delete_by_document_id(self, document_id: str) -> None:
        self.chunks = [chunk for chunk in self.chunks if chunk["document_id"] != document_id]


async def test_indexing_service_creates_embedded_chunks() -> None:
    vector_store = FakeVectorStore()
    service = IndexingService(
        splitter=TextSplitter(chunk_size=12, chunk_overlap=2),
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=vector_store,
    )
    document = Document(
        id="document-test",
        tenant_id="tenant-test",
        kb_id="kb-test",
        title="Test",
        content="first paragraph\n\nsecond paragraph",
    )

    count = await service.index_document(document)

    assert count == len(vector_store.chunks)
    assert count > 1
    assert all(chunk["document_id"] == "document-test" for chunk in vector_store.chunks)
    assert [chunk["metadata"]["chunk_index"] for chunk in vector_store.chunks] == list(range(count))


async def test_indexing_service_writes_chunks_to_vector_and_keyword_stores() -> None:
    vector_store = FakeVectorStore()
    keyword_provider = FakeKeywordSearchProvider()
    service = IndexingService(
        splitter=TextSplitter(chunk_size=100, chunk_overlap=10),
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=vector_store,
        keyword_search_provider=keyword_provider,
    )
    document = Document(
        id="document-test",
        tenant_id="tenant-test",
        kb_id="kb-test",
        title="Test",
        content="document content",
    )

    count = await service.index_document(document)

    assert count == len(vector_store.chunks) == len(keyword_provider.chunks)
    assert [chunk["id"] for chunk in vector_store.chunks] == [
        chunk["id"] for chunk in keyword_provider.chunks
    ]


async def test_indexing_service_uses_structured_splitter_for_markdown_documents() -> None:
    vector_store = FakeVectorStore()
    service = IndexingService(
        splitter=TextSplitter(chunk_size=100, chunk_overlap=10),
        markdown_splitter=MarkdownStructuredSplitter(
            chunk_size=100,
            chunk_overlap=10,
            table_max_rows_per_chunk=2,
        ),
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=vector_store,
    )
    document = Document(
        id="markdown-document",
        tenant_id="tenant-test",
        kb_id="kb-test",
        title="rules.md",
        content=(
            "# 交易规则\n\n"
            "退款说明。\n\n"
            "| 场景 | 说明 |\n"
            "| --- | --- |\n"
            "| 退款 | 原路退回 |\n"
        ),
    )

    count = await service.index_document(document)

    assert count == 2
    assert [chunk["metadata"]["chunk_type"] for chunk in vector_store.chunks] == [
        "section",
        "table",
    ]
    assert vector_store.chunks[0]["metadata"]["heading_path"] == "交易规则"
    assert vector_store.chunks[1]["metadata"]["heading_path"] == "交易规则"
    assert vector_store.chunks[1]["metadata"]["table_part"] == 1
