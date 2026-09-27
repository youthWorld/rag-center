"""Adapt legacy fixture data to the new storage scope contract (tests only)."""

import inspect


async def vector_scope(self, query_vector, *, tenant_id, index_versions, top_k):
    rows = []
    for kb, version in sorted(index_versions.items()):
        kwargs = {"tenant_id": tenant_id, "kb_id": kb, "top_k": top_k}
        if "index_version" in inspect.signature(self.similarity_search).parameters:
            kwargs["index_version"] = version
        chunks = await self.similarity_search(query_vector, **kwargs)
        rows.extend({**chunk, "kb_id": kb, "index_version": version} for chunk in chunks)
    if len(index_versions) > 1:
        rows.sort(key=lambda row: (-row.get("score", 0), row["kb_id"], row["chunk_id"]))
    return rows[:top_k]


async def keyword_scope(self, *, query, tenant_id, index_versions, top_k):
    rows = []
    for kb, version in sorted(index_versions.items()):
        kwargs = {"query": query, "tenant_id": tenant_id, "kb_id": kb, "top_k": top_k}
        if "index_version" in inspect.signature(self.keyword_search).parameters:
            kwargs["index_version"] = version
        chunks = await self.keyword_search(**kwargs)
        rows.extend({**chunk, "kb_id": kb, "index_version": version} for chunk in chunks)
    if len(index_versions) > 1:
        rows.sort(
            key=lambda row: (
                -row.get("bm25_score", row.get("score", 0)),
                row["kb_id"],
                row["chunk_id"],
            )
        )
    return rows[:top_k]
