from unittest.mock import AsyncMock

import pytest

from app.core.config import Settings
from app.providers.keyword_search.elasticsearch import ElasticsearchKeywordSearchProvider


class FakeIndices:
    def __init__(self) -> None:
        self.exists = AsyncMock(return_value=False)
        self.create = AsyncMock()


class FakeElasticsearchClient:
    def __init__(self) -> None:
        self.indices = FakeIndices()
        self.bulk = AsyncMock(return_value={"errors": False})
        self.search = AsyncMock(
            return_value={
                "hits": {
                    "hits": [
                        {
                            "_id": "chunk-1",
                            "_score": 12.4,
                            "_source": {
                                "tenant_id": "tenant-test",
                                "kb_id": "kb-test",
                                "document_id": "document-test",
                                "chunk_id": "chunk-1",
                                "title": "Refund policy",
                                "content": "Refunds are available.",
                            },
                        }
                    ]
                }
            }
        )
        self.delete_by_query = AsyncMock()
        self.close = AsyncMock()


@pytest.mark.asyncio
async def test_elasticsearch_provider_indexes_without_embedding_and_searches_with_filters() -> None:
    client = FakeElasticsearchClient()
    provider = ElasticsearchKeywordSearchProvider(
        Settings(elasticsearch_index="rag_chunks_test"),
        client=client,
    )

    await provider.add_chunks(
        [
            {
                "id": "chunk-1",
                "tenant_id": "tenant-test",
                "kb_id": "kb-test",
                "document_id": "document-test",
                "title": "Refund policy",
                "content": "Refunds are available.",
                "metadata": {"chunk_index": 0},
                "embedding": [0.1, 0.2],
            }
        ]
    )

    create_kwargs = client.indices.create.await_args.kwargs
    assert create_kwargs["index"] == "rag_chunks_test"
    assert create_kwargs["mappings"]["properties"]["title"]["analyzer"] == "ik_max_word"
    assert create_kwargs["mappings"]["properties"]["content"]["search_analyzer"] == "ik_smart"

    operations = client.bulk.await_args.kwargs["operations"]
    assert operations[0] == {"index": {"_index": "rag_chunks_test", "_id": "chunk-1"}}
    assert operations[1]["chunk_id"] == "chunk-1"
    assert "embedding" not in operations[1]

    results = await provider.keyword_search(
        query="refund",
        tenant_id="tenant-test",
        kb_id="kb-test",
        top_k=5,
    )

    query = client.search.await_args.kwargs["query"]
    assert query["bool"]["filter"] == [
        {"term": {"tenant_id": "tenant-test"}},
        {"term": {"kb_id": "kb-test"}},
    ]
    assert query["bool"]["must"]["multi_match"]["fields"] == ["title^2", "content"]
    assert results[0]["bm25_score"] == 12.4
