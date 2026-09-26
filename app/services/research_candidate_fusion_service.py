from __future__ import annotations

from collections import defaultdict
from typing import Any

from app.schemas.research import ResearchCandidate
from app.services.research_limits import MAX_RESEARCH_CANDIDATES
from app.services.retrieve_once_service import RetrieveOnceResult


class ResearchCandidateFusionService:
    def merge(self, results: list[RetrieveOnceResult]) -> list[ResearchCandidate]:
        merged: dict[tuple[str, str, str], ResearchCandidate] = {}
        content_keys: dict[tuple[str, str, str], tuple[str, str, str]] = {}
        by_aspect: dict[str, list[tuple[int, tuple[str, str, str]]]] = defaultdict(list)
        for result in results:
            if result.error is not None:
                continue
            task_candidates = self._task_candidates(result)
            seen_for_task: set[tuple[str, str, str]] = set()
            for rank, raw in enumerate(task_candidates, start=1):
                key = (raw.kb_id, raw.index_version, raw.chunk_id)
                normalized_content = " ".join(raw.content.split()).casefold()
                content_key = (raw.kb_id, raw.index_version, normalized_content)
                if normalized_content:
                    key = content_keys.setdefault(content_key, key)
                if key not in merged:
                    raw.best_rank = rank
                    raw.fusion_score = 1 / (60 + rank)
                    merged[key] = raw
                else:
                    candidate = merged[key]
                    candidate.aspect_ids = self._merge_values(
                        candidate.aspect_ids, raw.aspect_ids
                    )
                    candidate.query_ids = self._merge_values(candidate.query_ids, raw.query_ids)
                    candidate.rounds = sorted(set([*candidate.rounds, *raw.rounds]))
                    candidate.best_rank = min(candidate.best_rank or rank, rank)
                    if key not in seen_for_task:
                        candidate.fusion_score += 1 / (60 + rank)
                    sources = list(candidate.metadata.get("sources") or [candidate.source])
                    if raw.source not in sources:
                        sources.append(raw.source)
                    candidate.metadata["sources"] = sources
                    if raw.chunk_id != candidate.chunk_id:
                        duplicates = list(candidate.metadata.get("duplicate_chunk_ids") or [])
                        if raw.chunk_id not in duplicates:
                            duplicates.append(raw.chunk_id)
                        candidate.metadata["duplicate_chunk_ids"] = duplicates
                for aspect_id in result.aspect_ids:
                    by_aspect[aspect_id].append((rank, key))
                seen_for_task.add(key)

        selected: list[tuple[str, str, str]] = []
        selected_set: set[tuple[str, str, str]] = set()

        def add(key: tuple[str, str, str]) -> None:
            if (
                key in merged
                and key not in selected_set
                and len(selected) < MAX_RESEARCH_CANDIDATES
            ):
                selected.append(key)
                selected_set.add(key)

        for aspect_id in by_aspect:
            for _, key in sorted(by_aspect[aspect_id])[:1]:
                add(key)
        for slot in range(2):
            for aspect_id in by_aspect:
                ranked = sorted(by_aspect[aspect_id])
                if slot < len(ranked):
                    add(ranked[slot][1])
        for key, candidate in sorted(
            merged.items(), key=lambda item: (-len(item[1].aspect_ids), -item[1].fusion_score)
        ):
            if len(candidate.aspect_ids) > 1:
                add(key)
        for key, _ in sorted(merged.items(), key=lambda item: -item[1].fusion_score):
            add(key)
        output = [merged[key] for key in selected]
        for index, candidate in enumerate(output, start=1):
            candidate.candidate_id = f"C{index}"
        return output

    @classmethod
    def _task_candidates(cls, result: RetrieveOnceResult) -> list[ResearchCandidate]:
        candidates: list[ResearchCandidate] = []
        seen: set[tuple[str, str, str]] = set()
        for rank, chunk in enumerate(result.retrieved_chunks, start=1):
            raw = chunk.model_dump(mode="json")
            candidate = cls._from_chunk(raw, result=result, rank=rank)
            if candidate is not None:
                key = (candidate.kb_id, candidate.index_version, candidate.chunk_id)
                if key not in seen:
                    seen.add(key)
                    candidates.append(candidate)
            context = raw.get("context")
            sources = context.get("sources") if isinstance(context, dict) else None
            if isinstance(sources, list):
                for source in sources:
                    if not isinstance(source, dict) or source.get("relation") == "anchor":
                        continue
                    context_candidate = cls._from_context(source, anchor=raw, result=result)
                    if context_candidate is not None:
                        key = (
                            context_candidate.kb_id,
                            context_candidate.index_version,
                            context_candidate.chunk_id,
                        )
                        if key not in seen:
                            seen.add(key)
                            candidates.append(context_candidate)
        for rank, raw in enumerate(result.candidate_snapshot, start=1):
            candidate = cls._from_chunk(raw, result=result, rank=rank)
            if candidate is None:
                continue
            key = (candidate.kb_id, candidate.index_version, candidate.chunk_id)
            if key not in seen:
                seen.add(key)
                candidates.append(candidate)
        return candidates

    @staticmethod
    def _from_chunk(
        raw: dict[str, Any], *, result: RetrieveOnceResult, rank: int
    ) -> ResearchCandidate | None:
        kb_id = str(raw.get("kb_id") or "")
        version = str(raw.get("index_version") or "")
        chunk_id = str(raw.get("chunk_id") or "")
        document_id = str(raw.get("document_id") or "")
        if not all((kb_id, version, chunk_id, document_id)):
            return None
        metadata = dict(raw.get("metadata") or {})
        source = str(raw.get("retrieval_source") or "vector")
        if source == "graph":
            source = "graph_injection"
        return ResearchCandidate(
            chunk_id=chunk_id,
            kb_id=kb_id,
            document_id=document_id,
            index_version=version,
            title=str(raw.get("title") or ""),
            heading_path=metadata.get("heading_path"),
            content=str(raw.get("content") or ""),
            retrieval_text=raw.get("retrieval_text"),
            source=source,
            aspect_ids=list(result.aspect_ids),
            query_ids=[result.query_id],
            rounds=[result.round],
            best_rank=rank,
            metadata=metadata,
        )

    @staticmethod
    def _from_context(
        source: dict[str, Any], *, anchor: dict[str, Any], result: RetrieveOnceResult
    ) -> ResearchCandidate | None:
        chunk_id = str(source.get("chunk_id") or "")
        kb_id = str(anchor.get("kb_id") or "")
        version = str(anchor.get("index_version") or "")
        if not all((chunk_id, kb_id, version)):
            return None
        relation = str(source.get("relation") or "unknown")
        return ResearchCandidate(
            chunk_id=chunk_id,
            kb_id=kb_id,
            document_id=str(anchor.get("document_id") or ""),
            index_version=version,
            title=str(anchor.get("title") or ""),
            content=str(source.get("content") or ""),
            source=f"context:{relation}",
            context_relation=relation,
            aspect_ids=list(result.aspect_ids),
            query_ids=[result.query_id],
            rounds=[result.round],
            metadata={"anchor_chunk_id": anchor.get("chunk_id")},
        )

    @staticmethod
    def _merge_values(current: list[str], incoming: list[str]) -> list[str]:
        return list(dict.fromkeys([*current, *incoming]))
