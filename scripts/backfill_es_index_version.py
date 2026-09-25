"""Idempotently label pre-version Elasticsearch documents as v1.

Run after Alembic upgrade and before deploying version-filtered searches.
This does not reindex text or call an embedding provider.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from elasticsearch import AsyncElasticsearch  # noqa: E402

from app.core.config import settings  # noqa: E402


async def main() -> None:
    async with AsyncElasticsearch(settings.elasticsearch_url) as client:
        index = settings.elasticsearch_index
        if not await client.indices.exists(index=index):
            print(f"index {index} does not exist (nothing to backfill)")
            return
        await client.indices.put_mapping(
            index=index,
            properties={
                "index_version": {"type": "keyword"},
                "retrieval_text": {
                    "type": "text",
                    "analyzer": "ik_max_word",
                    "search_analyzer": "ik_smart",
                },
                "section_id": {"type": "keyword"},
                "parent_section_id": {"type": "keyword"},
                "order_index": {"type": "integer"},
            },
        )
        missing = {"bool": {"must_not": [{"exists": {"field": "index_version"}}]}}
        before = await client.count(index=index)
        response = await client.update_by_query(
            index=index,
            query=missing,
            script={"source": "ctx._source.index_version = 'v1'", "lang": "painless"},
            conflicts="proceed",
            refresh=True,
            wait_for_completion=True,
        )
        after = await client.count(index=index)
        remaining = await client.count(index=index, query=missing)
        if response.get("failures") or remaining["count"] or before["count"] != after["count"]:
            raise RuntimeError("ES v1 backfill incomplete; do not enable version-filtered search")
        print(f"index={index} documents={after['count']} updated={response['updated']} missing=0")


if __name__ == "__main__":
    asyncio.run(main())
