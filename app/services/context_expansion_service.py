from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.chunk import Chunk
from app.services.chunk_relation_resolver import ChunkRelationResolver

MAX_REFERENCE_TARGETS_PER_ANCHOR = 3
MAX_SUPPLEMENTAL_CHUNKS_PER_ANCHOR = 3
MAX_CONTEXT_CHARS_PER_ANCHOR = 4096


@dataclass(slots=True)
class ContextExpansionResult:
    anchors: list[dict[str, Any]] = field(default_factory=list)
    expanded_anchor_count: int = 0
    supplemental_chunk_count: int = 0
    deduplicated_count: int = 0
    latency_ms: int = 0
    degraded: bool = False
    error: str | None = None


class ContextExpansionService:
    """Attach one-hop context without changing anchor ranking or scores."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.logger = get_logger(__name__)
        self.relation_resolver = ChunkRelationResolver(session)

    async def expand(
        self,
        *,
        tenant_id: str,
        kb_ids: list[str],
        anchors: list[dict[str, Any]],
        index_version: str = "v2",
    ) -> ContextExpansionResult:
        started = time.perf_counter()
        result = ContextExpansionResult(anchors=[dict(anchor) for anchor in anchors])
        if not anchors:
            result.latency_ms = int((time.perf_counter() - started) * 1000)
            return result
        try:
            for index, anchor in enumerate(result.anchors):
                version = str(anchor.get("index_version") or index_version)
                if version == "v1":
                    continue
                supplemental, duplicates = await self._supplemental_chunks(
                    tenant_id=tenant_id,
                    kb_ids=kb_ids,
                    anchor=anchor,
                    index_version=version,
                )
                result.deduplicated_count += duplicates
                context = self._build_context(anchor, supplemental)
                if context["chunk_ids"] != [str(anchor.get("chunk_id"))]:
                    result.expanded_anchor_count += 1
                    result.supplemental_chunk_count += len(context["chunk_ids"]) - 1
                anchor["context"] = context
                result.anchors[index] = anchor
        except Exception as exc:
            result.degraded = True
            result.error = type(exc).__name__
            for anchor in result.anchors:
                anchor["context"] = None
            self.logger.warning(
                "BUSINESS_EVENT | event=context_expansion_degraded | tenant_id=%s | "
                "kb_ids=%s | error=%s",
                tenant_id,
                kb_ids,
                type(exc).__name__,
            )
        result.latency_ms = int((time.perf_counter() - started) * 1000)
        return result

    async def _supplemental_chunks(
        self,
        *,
        tenant_id: str,
        kb_ids: list[str],
        anchor: dict[str, Any],
        index_version: str,
    ) -> tuple[list[tuple[Chunk, str]], int]:
        kb_id = str(anchor.get("kb_id") or (kb_ids[0] if len(kb_ids) == 1 else ""))
        if kb_id not in kb_ids:
            return [], 0
        anchor_id = str(anchor.get("chunk_id") or "")
        related = await self.relation_resolver.resolve(
            tenant_id=tenant_id,
            kb_id=kb_id,
            index_version=index_version,
            anchor=anchor,
            reference_limit=MAX_REFERENCE_TARGETS_PER_ANCHOR,
        )
        deduped: list[tuple[Chunk, str]] = []
        seen_ids = {anchor_id}
        seen_content = {str(anchor.get("content") or "").strip()}
        duplicates = 0
        for related_item in related:
            chunk = related_item.chunk
            relation = related_item.relation
            if len(deduped) >= MAX_SUPPLEMENTAL_CHUNKS_PER_ANCHOR:
                break
            if chunk.id in seen_ids or chunk.content.strip() in seen_content:
                duplicates += 1
                continue
            seen_ids.add(chunk.id)
            seen_content.add(chunk.content.strip())
            deduped.append((chunk, relation))
        return deduped, duplicates

    @staticmethod
    def _build_context(
        anchor: dict[str, Any], supplemental: list[tuple[Chunk, str]]
    ) -> dict[str, Any]:
        anchor_id = str(anchor.get("chunk_id") or "")
        sources = [
            {
                "chunk_id": anchor_id,
                "relation": "anchor",
                "content": str(anchor.get("content") or ""),
            }
        ]
        chunk_ids = [anchor_id]
        selected: list[tuple[Chunk, str]] = []
        for chunk, relation in supplemental:
            proposed = [*selected, (chunk, relation)]
            if len(ContextExpansionService._compose_content(anchor, proposed)) > (
                MAX_CONTEXT_CHARS_PER_ANCHOR
            ):
                break
            selected.append((chunk, relation))
            chunk_ids.append(str(chunk.id))
            sources.append(
                {
                    "chunk_id": str(chunk.id),
                    "relation": relation,
                    "content": str(chunk.content or ""),
                }
            )
        return {
            "content": ContextExpansionService._compose_content(anchor, selected),
            "chunk_ids": chunk_ids,
            "sources": sources,
        }

    @staticmethod
    def _compose_content(
        anchor: dict[str, Any], supplemental: list[tuple[Chunk, str]]
    ) -> str:
        title = str(anchor.get("title") or "").strip()
        heading_path = str((anchor.get("metadata") or {}).get("heading_path") or "").strip()
        content_parts: list[str] = []
        if title:
            content_parts.append(f"文档：{title}")
        if heading_path:
            content_parts.append(f"章节：{heading_path}")
        content_parts.extend(
            str(chunk.content or "")
            for chunk, relation in supplemental
            if relation == "parent_section"
        )
        content_parts.extend(
            str(chunk.content or "")
            for chunk, relation in supplemental
            if relation == "previous"
        )
        content_parts.append(str(anchor.get("content") or ""))
        content_parts.extend(
            str(chunk.content or "")
            for chunk, relation in supplemental
            if relation == "next"
        )
        references = [
            str(chunk.content or "")
            for chunk, relation in supplemental
            if relation == "reference"
        ]
        if references:
            content_parts.append("引用内容：\n" + "\n\n".join(references))
        return "\n\n".join(part for part in content_parts if part)
