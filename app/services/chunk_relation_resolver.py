from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chunk import Chunk
from app.models.chunk_relation import ChunkRelation

RELATION_PRIORITY = {
    "reference": 0,
    "parent_section": 1,
    "previous": 2,
    "next": 3,
}


@dataclass(frozen=True, slots=True)
class ResolvedChunkRelation:
    chunk: Chunk
    relation: str


class ChunkRelationResolver:
    """Resolve one-hop relations with one shared permission and ordering policy."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def resolve(
        self,
        *,
        tenant_id: str,
        kb_id: str,
        index_version: str,
        anchor: dict[str, Any],
        reference_limit: int | None = None,
    ) -> list[ResolvedChunkRelation]:
        related: list[ResolvedChunkRelation] = []
        related.extend(
            await self._resolve_references(
                tenant_id=tenant_id,
                kb_id=kb_id,
                index_version=index_version,
                anchor=anchor,
                limit=reference_limit,
            )
        )
        parent = await self._resolve_parent(
            tenant_id=tenant_id,
            kb_id=kb_id,
            index_version=index_version,
            anchor=anchor,
        )
        if parent is not None:
            related.append(parent)
        related.extend(
            await self._resolve_neighbors(
                tenant_id=tenant_id,
                kb_id=kb_id,
                index_version=index_version,
                anchor=anchor,
            )
        )
        related.sort(
            key=lambda item: (
                RELATION_PRIORITY[item.relation],
                item.chunk.order_index is None,
                item.chunk.order_index if item.chunk.order_index is not None else 0,
                str(item.chunk.id),
            )
        )
        return related

    async def _resolve_references(
        self,
        *,
        tenant_id: str,
        kb_id: str,
        index_version: str,
        anchor: dict[str, Any],
        limit: int | None,
    ) -> list[ResolvedChunkRelation]:
        query = (
            select(ChunkRelation, Chunk)
            .join(Chunk, Chunk.id == ChunkRelation.target_chunk_id)
            .where(
                ChunkRelation.tenant_id == tenant_id,
                ChunkRelation.kb_id == kb_id,
                ChunkRelation.index_version == index_version,
                ChunkRelation.source_chunk_id == str(anchor.get("chunk_id") or ""),
                ChunkRelation.relation_type == "reference",
                Chunk.tenant_id == tenant_id,
                Chunk.kb_id == kb_id,
                Chunk.index_version == index_version,
            )
            .order_by(Chunk.order_index.asc(), Chunk.id.asc())
        )
        if limit is not None:
            query = query.limit(limit)
        result = await self.session.execute(query)
        return [
            ResolvedChunkRelation(chunk=chunk, relation="reference")
            for _, chunk in result.all()
        ]

    async def _resolve_parent(
        self,
        *,
        tenant_id: str,
        kb_id: str,
        index_version: str,
        anchor: dict[str, Any],
    ) -> ResolvedChunkRelation | None:
        document_id = str(anchor.get("document_id") or "")
        parent_section_id = anchor.get("parent_section_id")
        order_index = anchor.get("order_index")
        if not document_id or not parent_section_id or not isinstance(order_index, int):
            return None
        query = (
            select(Chunk)
            .where(
                Chunk.tenant_id == tenant_id,
                Chunk.kb_id == kb_id,
                Chunk.document_id == document_id,
                Chunk.index_version == index_version,
                Chunk.section_id == str(parent_section_id),
                Chunk.order_index < order_index,
            )
            .order_by(Chunk.order_index.desc(), Chunk.id.desc())
            .limit(1)
        )
        chunk = (await self.session.execute(query)).scalars().first()
        return (
            ResolvedChunkRelation(chunk=chunk, relation="parent_section")
            if chunk is not None
            else None
        )

    async def _resolve_neighbors(
        self,
        *,
        tenant_id: str,
        kb_id: str,
        index_version: str,
        anchor: dict[str, Any],
    ) -> list[ResolvedChunkRelation]:
        metadata = anchor.get("metadata") or {}
        if metadata.get("chunk_type") != "prose":
            return []
        document_id = str(anchor.get("document_id") or "")
        order_index = anchor.get("order_index")
        section_id = anchor.get("section_id")
        if not document_id or not isinstance(order_index, int) or not section_id:
            return []
        query = (
            select(Chunk)
            .where(
                Chunk.tenant_id == tenant_id,
                Chunk.kb_id == kb_id,
                Chunk.document_id == document_id,
                Chunk.index_version == index_version,
                Chunk.section_id == section_id,
                Chunk.order_index.in_([order_index - 1, order_index + 1]),
            )
            .order_by(Chunk.order_index.asc())
            .limit(2)
        )
        chunks = (await self.session.execute(query)).scalars().all()
        return [
            ResolvedChunkRelation(
                chunk=chunk,
                relation="previous" if chunk.order_index == order_index - 1 else "next",
            )
            for chunk in chunks
        ]
