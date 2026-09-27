from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.chunk import Chunk
from app.services.chunk_relation_resolver import RELATION_PRIORITY, ChunkRelationResolver
from app.utils.markdown_splitter import is_low_information_content, normalize_content

MAX_GRAPH_SEEDS = 10
MAX_INJECTED_CHUNKS_PER_SEED = 3
MAX_TOTAL_CANDIDATES = 30


@dataclass(slots=True)
class GraphCandidateExpansionResult:
    candidates: list[dict[str, Any]] = field(default_factory=list)
    sources: dict[str, list[dict[str, str]]] = field(default_factory=dict)
    injected_count: int = 0
    latency_ms: int = 0
    degraded: bool = False
    error: str | None = None


class GraphCandidateExpansionService:
    """Expand Hybrid seeds along deterministic, one-hop document relations."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.logger = get_logger(__name__)
        self.relation_resolver = ChunkRelationResolver(session)

    async def expand_candidates(
        self,
        *,
        tenant_id: str,
        kb_ids: list[str],
        seeds: list[dict[str, Any]],
        index_version: str = "v2",
        max_injected: int | None = None,
    ) -> GraphCandidateExpansionResult:
        started = time.perf_counter()
        result = GraphCandidateExpansionResult()
        if not seeds:
            return result

        existing_ids = {(str(seed.get("kb_id") or ""), str(seed.get("chunk_id"))) for seed in seeds}
        seen_content = {
            (str(seed.get("kb_id") or ""), normalize_content(str(seed.get("content") or "")))
            for seed in seeds
        }
        limit = max(0, MAX_TOTAL_CANDIDATES - len(seeds)) if max_injected is None else max_injected
        try:
            for seed_index, seed in enumerate(seeds[:MAX_GRAPH_SEEDS]):
                seed_id = str(seed.get("chunk_id") or "")
                if not seed_id:
                    continue
                kb_id = str(seed.get("kb_id") or (kb_ids[0] if len(kb_ids) == 1 else ""))
                if kb_id not in kb_ids:
                    continue
                version = str(seed.get("index_version") or index_version)
                if version == "v1":
                    continue
                related = await self._related_chunks(
                    tenant_id=tenant_id,
                    kb_id=kb_id,
                    index_version=version,
                    seed=seed,
                )
                added_for_seed = 0
                for related_item in related:
                    chunk = related_item.chunk
                    relation = related_item.relation
                    chunk_id = str(chunk.id)
                    identity = (kb_id, chunk_id)
                    if identity in existing_ids:
                        continue
                    if max_injected is not None:
                        normalized = normalize_content(chunk.content)
                        if (
                            (kb_id, normalized) in seen_content
                            or (chunk.chunk_metadata or {}).get("chunk_type") == "reference_pointer"
                            or is_low_information_content(chunk.content)
                        ):
                            continue
                    if len(result.candidates) >= limit:
                        break
                    if added_for_seed >= MAX_INJECTED_CHUNKS_PER_SEED:
                        break
                    item = self._chunk_to_dict(chunk)
                    item["kb_id"] = kb_id
                    item["kb_name"] = seed.get("kb_name")
                    item["injection_source"] = relation
                    item["metadata"] = {
                        **item.get("metadata", {}),
                        "injection_source": relation,
                        "injection_anchor_id": seed_id,
                    }
                    item["_graph_anchor_rank"] = seed_index
                    item["_graph_anchor_id"] = seed_id
                    item["_graph_relation_rank"] = RELATION_PRIORITY[relation]
                    result.candidates.append(item)
                    existing_ids.add(identity)
                    seen_content.add((kb_id, normalize_content(chunk.content)))
                    result.sources.setdefault(seed_id, []).append(
                        {"chunk_id": chunk_id, "relation": relation}
                    )
                    added_for_seed += 1
                if len(result.candidates) >= limit:
                    break
        except Exception as exc:
            result.degraded = True
            result.error = type(exc).__name__
            self.logger.warning(
                "BUSINESS_EVENT | event=graph_candidate_expansion_degraded | "
                "tenant_id=%s | kb_ids=%s | error=%s",
                tenant_id,
                kb_ids,
                type(exc).__name__,
            )
            result.candidates.clear()
            result.sources.clear()
        result.injected_count = len(result.candidates)
        result.latency_ms = int((time.perf_counter() - started) * 1000)
        return result

    async def _related_chunks(
        self,
        *,
        tenant_id: str,
        kb_id: str,
        index_version: str,
        seed: dict[str, Any],
    ) -> list[Any]:
        return await self.relation_resolver.resolve(
            tenant_id=tenant_id,
            kb_id=kb_id,
            index_version=index_version,
            anchor=seed,
            reference_limit=MAX_INJECTED_CHUNKS_PER_SEED,
        )

    @staticmethod
    def _chunk_to_dict(chunk: Chunk) -> dict[str, Any]:
        return {
            "document_id": chunk.document_id,
            "chunk_id": chunk.id,
            "title": chunk.title,
            "content": chunk.content,
            "retrieval_text": chunk.retrieval_text,
            "index_version": chunk.index_version,
            "section_id": chunk.section_id,
            "parent_section_id": chunk.parent_section_id,
            "order_index": chunk.order_index,
            "score": 0.0,
            "vector_score": None,
            "bm25_score": None,
            "vector_rank": None,
            "bm25_rank": None,
            "retrieval_source": "graph",
            "metadata": dict(chunk.chunk_metadata or {}),
        }
