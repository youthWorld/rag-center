from app.models.document import Document
from app.providers.embedding.base import EmbeddingProvider
from app.providers.vectorstores.base import VectorStore
from app.services.indexing_service import IndexingService
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
