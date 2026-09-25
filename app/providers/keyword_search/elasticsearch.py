from datetime import UTC, datetime
from typing import Any

from elasticsearch import AsyncElasticsearch
from elasticsearch.exceptions import ConflictError

from app.core.config import Settings
from app.providers.keyword_search.base import KeywordSearchProvider


class ElasticsearchKeywordSearchProvider(KeywordSearchProvider):
    """Store chunk text in Elasticsearch and retrieve it with BM25."""

    INDEX_MAPPING = {
        "properties": {
            "tenant_id": {"type": "keyword"},
            "kb_id": {"type": "keyword"},
            "document_id": {"type": "keyword"},
            "chunk_id": {"type": "keyword"},
            "index_version": {"type": "keyword"},
            "title": {
                "type": "text",
                "analyzer": "ik_max_word",
                "search_analyzer": "ik_smart",
            },
            "content": {
                "type": "text",
                "analyzer": "ik_max_word",
                "search_analyzer": "ik_smart",
            },
            "retrieval_text": {
                "type": "text",
                "analyzer": "ik_max_word",
                "search_analyzer": "ik_smart",
            },
            "section_id": {"type": "keyword"},
            "parent_section_id": {"type": "keyword"},
            "order_index": {"type": "integer"},
            "metadata": {"type": "object", "enabled": False},
            "created_at": {"type": "date"},
        }
    }

    def __init__(
        self,
        settings: Settings,
        client: AsyncElasticsearch | Any | None = None,
    ) -> None:
        self.settings = settings
        self.index = settings.elasticsearch_index
        self.client = client or AsyncElasticsearch(settings.elasticsearch_url)
        self._index_ready = False

    async def add_chunks(self, chunks: list[dict[str, Any]]) -> None:
        if not chunks:
            return

        await self._ensure_index()
        operations: list[dict[str, Any]] = []
        for chunk in chunks:
            chunk_id = str(chunk.get("chunk_id") or chunk.get("id") or "")
            if not chunk_id:
                raise ValueError("chunk must contain an id")

            operations.append({"index": {"_index": self.index, "_id": chunk_id}})
            operations.append(
                {
                    "tenant_id": chunk["tenant_id"],
                    "kb_id": chunk["kb_id"],
                    "document_id": chunk["document_id"],
                    "chunk_id": chunk_id,
                    "index_version": str(chunk.get("index_version") or "v1"),
                    "title": chunk["title"],
                    "content": chunk["content"],
                    "retrieval_text": chunk.get("retrieval_text"),
                    "section_id": chunk.get("section_id"),
                    "parent_section_id": chunk.get("parent_section_id"),
                    "order_index": chunk.get("order_index"),
                    "metadata": chunk.get("metadata", {}),
                    "created_at": self._serialize_created_at(chunk.get("created_at")),
                }
            )

        response = await self.client.bulk(operations=operations, refresh="wait_for")
        body = self._response_body(response)
        if body.get("errors"):
            raise RuntimeError("elasticsearch bulk indexing failed")

    async def keyword_search(
        self,
        *,
        query: str,
        tenant_id: str,
        kb_id: str,
        top_k: int = 20,
        index_version: str | None = None,
    ) -> list[dict[str, Any]]:
        if top_k < 1:
            return []

        await self._ensure_index()
        filters = [
            {"term": {"tenant_id": tenant_id}},
            {"term": {"kb_id": kb_id}},
        ]
        if index_version is not None:
            filters.append({"term": {"index_version": index_version}})
        fields = ["title^2", "content"]
        if index_version and index_version != "v1":
            fields = ["retrieval_text", "title^2"]
        query_body = {
            "bool": {
                "filter": filters,
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": fields,
                    }
                },
            }
        }
        response = await self.client.search(
            index=self.index,
            query=query_body,
            size=top_k,
        )
        body = self._response_body(response)
        hits = body.get("hits", {}).get("hits", [])
        results: list[dict[str, Any]] = []
        for hit in hits:
            source = dict(hit.get("_source") or {})
            chunk_id = source.get("chunk_id") or hit.get("_id")
            if chunk_id is None:
                continue
            bm25_score = float(hit.get("_score") or 0.0)
            results.append(
                {
                    **source,
                    "chunk_id": str(chunk_id),
                    "bm25_score": bm25_score,
                }
            )
        return results

    async def delete_by_document_id(
        self, document_id: str, *, index_version: str | None = None
    ) -> None:
        if not await self._index_exists():
            return
        filters: list[dict[str, Any]] = [{"term": {"document_id": document_id}}]
        if index_version is not None:
            filters.append({"term": {"index_version": index_version}})
        await self.client.delete_by_query(
            index=self.index,
            query={"bool": {"filter": filters}},
            conflicts="proceed",
            refresh=True,
        )

    async def close(self) -> None:
        await self.client.close()

    async def _ensure_index(self) -> None:
        if self._index_ready:
            return
        if not await self._index_exists():
            try:
                await self.client.indices.create(
                    index=self.index,
                    mappings=self.INDEX_MAPPING,
                )
            except ConflictError:
                pass
        self._index_ready = True

    async def _index_exists(self) -> bool:
        response = await self.client.indices.exists(index=self.index)
        return self._response_bool(response)

    @staticmethod
    def _serialize_created_at(value: Any) -> str:
        if isinstance(value, datetime):
            return value.astimezone(UTC).isoformat()
        if value:
            return str(value)
        return datetime.now(UTC).isoformat()

    @staticmethod
    def _response_body(response: Any) -> dict[str, Any]:
        body = getattr(response, "body", response)
        return body if isinstance(body, dict) else {}

    @classmethod
    def _response_bool(cls, response: Any) -> bool:
        if isinstance(response, bool):
            return response
        body = getattr(response, "body", response)
        if isinstance(body, bool):
            return body
        if isinstance(body, dict):
            if "value" in body:
                return bool(body["value"])
            return bool(body)
        status = getattr(getattr(response, "meta", None), "status", None)
        if status is not None:
            return 200 <= status < 300
        return bool(response)
