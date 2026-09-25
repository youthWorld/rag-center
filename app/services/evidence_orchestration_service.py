from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.error_codes import ErrorCode
from app.core.exceptions import LLMServiceError, ServiceConfigurationError
from app.core.logging import get_logger
from app.providers.llm.base import LLMProvider, LLMProviderError
from app.repositories.chunk_repository import ChunkRepository
from app.schemas.rag import (
    EvidenceGroup,
    EvidenceGroupItem,
    EvidenceItem,
    EvidenceOptions,
    EvidencePack,
    RetrieveEvidenceMetadata,
)
from app.utils.markdown_splitter import normalize_content

MAX_EVIDENCE_CANDIDATES = 20
TOP_K_TARGET = 10
CONTEXT_SOURCE_TARGET = 4
BASE_POOL_TARGET = 6
DEFAULT_MAX_EVIDENCE_ITEMS = 8
MAX_EVIDENCE_TEXT_CHARS = 2048
MAX_EVIDENCE_OUTPUT_CHARS = 12000
MAX_ASPECT_CHARS = 100

EVIDENCE_ORCHESTRATION_PROMPT = """你是 RAG 证据编排器，不负责回答问题。

任务：
1. 将用户问题拆成回答时必须覆盖的最少要点。
2. 只从给定 candidates 中选择能够支撑这些要点的证据。
3. core 表示直接回答要点，supporting 表示提供必要条件、范围或补充说明。
4. 不要因为主题相似就选择候选，不要选择重复证据。
5. 每个必要要点都必须输出；没有核心证据时返回空 evidence 数组。
6. 同一候选可以支撑多个要点。
7. 不生成答案，不改写证据，不输出候选中不存在的事实。

严格返回 JSON：
{
  \"aspects\": [
    {
      \"aspect\": \"简短的问题要点\",
      \"evidence\": [
        {
          \"index\": 0,
          \"role\": \"core|supporting\"
        }
      ]
    }
  ]
}"""


@dataclass(slots=True)
class EvidenceOrchestrationResult:
    pack: EvidencePack | None
    metadata: RetrieveEvidenceMetadata


class EvidenceValidationError(ValueError):
    pass


class EvidenceOrchestrationService:
    def __init__(
        self,
        session: AsyncSession,
        llm_provider: LLMProvider,
        *,
        timeout_seconds: int = 15,
        chunk_repository: ChunkRepository | None = None,
    ) -> None:
        self.llm_provider = llm_provider
        self.timeout_seconds = timeout_seconds
        self.chunk_repository = chunk_repository or ChunkRepository(session)
        self.logger = get_logger(__name__)

    async def orchestrate(
        self,
        *,
        query: str,
        tenant_id: str,
        kb_ids: list[str],
        index_versions: dict[str, str],
        candidates: list[dict[str, Any]],
        retrieved_chunks: list[dict[str, Any]],
        options: EvidenceOptions,
    ) -> EvidenceOrchestrationResult:
        metadata = RetrieveEvidenceMetadata(enabled=bool(options.enabled))
        if not metadata.enabled:
            return EvidenceOrchestrationResult(pack=None, metadata=metadata)

        started = time.perf_counter()
        try:
            pool = self._build_candidate_pool(
                kb_ids=kb_ids,
                index_versions=index_versions,
                candidates=candidates,
                retrieved_chunks=retrieved_chunks,
            )
            metadata.candidate_count = len(pool)
            metadata.context_sources_included = any(
                str(candidate["source"]).startswith("context:") for candidate in pool
            )
            if not pool:
                return EvidenceOrchestrationResult(pack=None, metadata=metadata)

            payload = {
                "question": query,
                "max_evidence_items": options.max_items or DEFAULT_MAX_EVIDENCE_ITEMS,
                "candidates": [
                    {
                        "index": index,
                        "title": candidate["title"],
                        "heading_path": candidate.get("heading_path"),
                        "text": str(candidate["text"])[:MAX_EVIDENCE_TEXT_CHARS],
                        "retrieved_rank": candidate.get("retrieved_rank"),
                        "source": candidate["source"],
                    }
                    for index, candidate in enumerate(pool)
                ],
            }
            candidate_chars = sum(len(item["text"]) for item in payload["candidates"])
            metadata.executed = True
            self.logger.info(
                "EVIDENCE_LLM_REQUEST | model_candidates=%s | candidate_chars=%s",
                len(pool),
                candidate_chars,
            )
            llm_result = await asyncio.wait_for(
                self.llm_provider.chat_json_with_metadata(
                    system_prompt=EVIDENCE_ORCHESTRATION_PROMPT,
                    user_payload=payload,
                    temperature=0.0,
                    timeout_seconds=self.timeout_seconds,
                    max_tokens=2048,
                    enable_thinking=False,
                ),
                timeout=self.timeout_seconds,
            )
            metadata.model_call = llm_result.metadata.to_dict()
            aspects, selected_indexes = self._validate_output(
                llm_result.output,
                candidate_count=len(pool),
                max_items=options.max_items or DEFAULT_MAX_EVIDENCE_ITEMS,
            )
            selected_candidates = [pool[index] for index in selected_indexes]
            reloaded = await self._reload_selected_chunks(
                tenant_id=tenant_id,
                kb_ids=kb_ids,
                index_versions=index_versions,
                selected=selected_candidates,
            )
            pack, budget_exceeded = self._build_pack(
                aspects=aspects,
                pool=pool,
                selected_indexes=selected_indexes,
                reloaded=reloaded,
            )
            metadata.evidence_count = len(pack.items)
            metadata.output_chars = sum(len(item.content) for item in pack.items)
            metadata.budget_exceeded = budget_exceeded
            metadata.status = pack.status
            self.logger.info(
                "EVIDENCE_LLM_RESPONSE | candidate_count=%s | evidence_count=%s | status=%s",
                len(pool),
                len(pack.items),
                pack.status,
            )
            return EvidenceOrchestrationResult(pack=pack, metadata=metadata)
        except Exception as exc:
            metadata.degraded = True
            metadata.error_code, metadata.error = self._safe_error(exc)
            self.logger.warning(
                "BUSINESS_EVENT | event=evidence_orchestration_degraded | "
                "tenant_id=%s | kb_count=%s | candidate_count=%s | "
                "error_code=%s | error=%s",
                tenant_id,
                len(kb_ids),
                metadata.candidate_count,
                metadata.error_code,
                metadata.error,
            )
            return EvidenceOrchestrationResult(pack=None, metadata=metadata)
        finally:
            metadata.latency_ms = int((time.perf_counter() - started) * 1000)

    @classmethod
    def _build_candidate_pool(
        cls,
        *,
        kb_ids: list[str],
        index_versions: dict[str, str],
        candidates: list[dict[str, Any]],
        retrieved_chunks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        top: list[dict[str, Any]] = []
        contexts: list[dict[str, Any]] = []
        base: list[dict[str, Any]] = []
        top_ids: set[str] = set()

        for rank, chunk in enumerate(retrieved_chunks, start=1):
            candidate = cls._candidate_from_chunk(
                chunk,
                kb_ids=kb_ids,
                index_versions=index_versions,
                source=cls._runtime_source(chunk.get("retrieval_source")),
                retrieved_rank=rank,
            )
            if candidate is None:
                continue
            top.append(candidate)
            top_ids.add(candidate["chunk_id"])
            context = chunk.get("context")
            sources = context.get("sources") if isinstance(context, dict) else None
            if not isinstance(sources, list):
                continue
            for source in sources:
                if not isinstance(source, dict) or source.get("relation") == "anchor":
                    continue
                context_candidate = cls._candidate_from_context(
                    source, anchor=chunk, kb_ids=kb_ids, index_versions=index_versions
                )
                if context_candidate is not None:
                    contexts.append(context_candidate)

        for chunk in candidates:
            if str(chunk.get("chunk_id") or "") in top_ids:
                continue
            candidate = cls._candidate_from_chunk(
                chunk,
                kb_ids=kb_ids,
                index_versions=index_versions,
                source=cls._runtime_source(chunk.get("retrieval_source")),
                retrieved_rank=None,
            )
            if candidate is not None:
                base.append(candidate)

        top = cls._deduplicate(top)
        contexts = cls._deduplicate(contexts)
        base = cls._deduplicate(base)

        selected: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        seen_content: set[str] = set()

        def add_from(source: list[dict[str, Any]], limit: int | None = None) -> None:
            added = 0
            for candidate in source:
                if len(selected) >= MAX_EVIDENCE_CANDIDATES:
                    return
                chunk_id = candidate["chunk_id"]
                content = normalize_content(candidate["content"])
                if chunk_id in seen_ids or (content and content in seen_content):
                    continue
                selected.append(candidate)
                seen_ids.add(chunk_id)
                if content:
                    seen_content.add(content)
                added += 1
                if limit is not None and added >= limit:
                    return

        # Reserve each category's target before redistributing unused slots.
        # Cross-category duplicates are skipped while scanning, so a duplicate
        # cannot silently consume the six outside-TopK base reservations.
        add_from(top, TOP_K_TARGET)
        add_from(contexts, CONTEXT_SOURCE_TARGET)
        add_from(base, BASE_POOL_TARGET)
        if len(selected) < MAX_EVIDENCE_CANDIDATES:
            add_from(top)
            add_from(contexts)
            add_from(base)
        return selected

    @staticmethod
    def _candidate_from_chunk(
        chunk: dict[str, Any],
        *,
        kb_ids: list[str],
        index_versions: dict[str, str],
        source: str,
        retrieved_rank: int | None,
    ) -> dict[str, Any] | None:
        kb_id = str(chunk.get("kb_id") or (kb_ids[0] if len(kb_ids) == 1 else ""))
        chunk_id = str(chunk.get("chunk_id") or "")
        if not kb_id or kb_id not in kb_ids or not chunk_id:
            return None
        index_version = str(chunk.get("index_version") or index_versions.get(kb_id) or "")
        if not index_version or index_versions.get(kb_id) != index_version:
            return None
        metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
        content = str(chunk.get("content") or "")
        return {
            "chunk_id": chunk_id,
            "kb_id": kb_id,
            "index_version": index_version,
            "title": str(chunk.get("title") or ""),
            "heading_path": metadata.get("heading_path"),
            "content": content,
            "text": str(chunk.get("retrieval_text") or content),
            "source": source,
            "retrieved_rank": retrieved_rank,
        }

    @classmethod
    def _candidate_from_context(
        cls,
        source: dict[str, Any],
        *,
        anchor: dict[str, Any],
        kb_ids: list[str],
        index_versions: dict[str, str],
    ) -> dict[str, Any] | None:
        relation = str(source.get("relation") or "")
        chunk_id = str(source.get("chunk_id") or "")
        kb_id = str(anchor.get("kb_id") or (kb_ids[0] if len(kb_ids) == 1 else ""))
        index_version = str(anchor.get("index_version") or index_versions.get(kb_id) or "")
        if (
            not relation
            or not chunk_id
            or not kb_id
            or kb_id not in kb_ids
            or index_versions.get(kb_id) != index_version
        ):
            return None
        content = str(source.get("content") or "")
        return {
            "chunk_id": chunk_id,
            "kb_id": kb_id,
            "index_version": index_version,
            "title": str(anchor.get("title") or ""),
            "heading_path": None,
            "content": content,
            "text": content,
            "source": f"context:{relation}",
            "retrieved_rank": None,
            "anchor_chunk_id": str(anchor.get("chunk_id") or ""),
        }

    @staticmethod
    def _runtime_source(value: Any) -> str:
        source = str(value or "vector")
        return "graph_injection" if source == "graph" else source

    @staticmethod
    def _deduplicate(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        seen_content: set[str] = set()
        for candidate in candidates:
            chunk_id = candidate["chunk_id"]
            content = normalize_content(candidate["content"])
            if chunk_id in seen_ids or (content and content in seen_content):
                continue
            seen_ids.add(chunk_id)
            if content:
                seen_content.add(content)
            result.append(candidate)
        return result

    @staticmethod
    def _validate_output(
        payload: dict[str, Any], *, candidate_count: int, max_items: int
    ) -> tuple[list[dict[str, Any]], list[int]]:
        aspects = payload.get("aspects") if isinstance(payload, dict) else None
        if not isinstance(aspects, list) or not aspects:
            raise EvidenceValidationError("invalid aspects")
        normalized: list[dict[str, Any]] = []
        selected: list[int] = []
        selected_set: set[int] = set()
        for raw_aspect in aspects:
            if not isinstance(raw_aspect, dict):
                raise EvidenceValidationError("invalid aspect item")
            aspect = raw_aspect.get("aspect")
            if (
                not isinstance(aspect, str)
                or not aspect.strip()
                or len(aspect.strip()) > MAX_ASPECT_CHARS
            ):
                raise EvidenceValidationError("invalid aspect text")
            evidence = raw_aspect.get("evidence")
            if not isinstance(evidence, list):
                raise EvidenceValidationError("invalid evidence list")
            seen_in_aspect: set[int] = set()
            normalized_evidence: list[dict[str, Any]] = []
            for reference in evidence:
                if not isinstance(reference, dict):
                    raise EvidenceValidationError("invalid evidence reference")
                index = reference.get("index")
                role = reference.get("role")
                if (
                    not isinstance(index, int)
                    or isinstance(index, bool)
                    or index < 0
                    or index >= candidate_count
                ):
                    raise EvidenceValidationError("evidence index out of range")
                if index in seen_in_aspect:
                    raise EvidenceValidationError("duplicate evidence index")
                if role not in {"core", "supporting"}:
                    raise EvidenceValidationError("invalid evidence role")
                seen_in_aspect.add(index)
                normalized_evidence.append({"index": index, "role": role})
                if index not in selected_set:
                    selected_set.add(index)
                    selected.append(index)
            normalized.append({"aspect": aspect.strip(), "evidence": normalized_evidence})
        if len(selected) > max_items:
            raise EvidenceValidationError("evidence item limit exceeded")
        return normalized, selected

    async def _reload_selected_chunks(
        self,
        *,
        tenant_id: str,
        kb_ids: list[str],
        index_versions: dict[str, str],
        selected: list[dict[str, Any]],
    ) -> dict[tuple[str, str, str], Any]:
        scopes: dict[tuple[str, str], set[str]] = {}
        for candidate in selected:
            kb_id = candidate["kb_id"]
            version = candidate["index_version"]
            if kb_id not in kb_ids or index_versions.get(kb_id) != version:
                raise EvidenceValidationError("candidate scope mismatch")
            scopes.setdefault((kb_id, version), set()).add(candidate["chunk_id"])
        chunks = await self.chunk_repository.get_by_scopes(tenant_id=tenant_id, scopes=scopes)
        reloaded = {(chunk.kb_id, chunk.index_version, chunk.id): chunk for chunk in chunks}
        expected = {
            (candidate["kb_id"], candidate["index_version"], candidate["chunk_id"])
            for candidate in selected
        }
        if set(reloaded) != expected:
            raise EvidenceValidationError("selected chunks failed scope validation")
        return reloaded

    @staticmethod
    def _build_pack(
        *,
        aspects: list[dict[str, Any]],
        pool: list[dict[str, Any]],
        selected_indexes: list[int],
        reloaded: dict[tuple[str, str, str], Any],
    ) -> tuple[EvidencePack, bool]:
        evidence_ids: dict[int, str] = {}
        items: list[EvidenceItem] = []
        output_chars = 0
        budget_exceeded = False
        for index in selected_indexes:
            candidate = pool[index]
            chunk = reloaded[
                (candidate["kb_id"], candidate["index_version"], candidate["chunk_id"])
            ]
            content = str(chunk.content or "")
            if output_chars + len(content) > MAX_EVIDENCE_OUTPUT_CHARS:
                budget_exceeded = True
                continue
            evidence_id = f"E{len(items) + 1}"
            evidence_ids[index] = evidence_id
            output_chars += len(content)
            metadata = chunk.chunk_metadata if isinstance(chunk.chunk_metadata, dict) else {}
            items.append(
                EvidenceItem(
                    evidence_id=evidence_id,
                    chunk_id=chunk.id,
                    kb_id=chunk.kb_id,
                    document_id=chunk.document_id,
                    title=chunk.title,
                    heading_path=metadata.get("heading_path"),
                    index_version=chunk.index_version,
                    content=content,
                    source=candidate["source"],
                    retrieved_rank=candidate.get("retrieved_rank"),
                )
            )

        groups: list[EvidenceGroup] = []
        missing: list[str] = []
        for aspect in aspects:
            references = [
                EvidenceGroupItem(
                    evidence_id=evidence_ids[item["index"]],
                    role=item["role"],
                )
                for item in aspect["evidence"]
                if item["index"] in evidence_ids
            ]
            covered = any(reference.role == "core" for reference in references)
            if not covered:
                missing.append(aspect["aspect"])
            groups.append(
                EvidenceGroup(
                    aspect=aspect["aspect"],
                    evidence=references,
                    covered=covered,
                )
            )
        covered_count = sum(group.covered for group in groups)
        status = (
            "complete"
            if covered_count == len(groups)
            else "missing"
            if covered_count == 0
            else "partial"
        )
        return (
            EvidencePack(status=status, missing_aspects=missing, groups=groups, items=items),
            budget_exceeded,
        )

    @staticmethod
    def _safe_error(exception: Exception) -> tuple[str, str]:
        if isinstance(exception, EvidenceValidationError):
            return "EVIDENCE_VALIDATION_ERROR", str(exception)
        if isinstance(exception, TimeoutError):
            return ErrorCode.LLM_TIMEOUT.name, ErrorCode.LLM_TIMEOUT.message
        if isinstance(exception, LLMServiceError):
            error_code = exception.error_code or ErrorCode.LLM_ERROR
            return error_code.name, error_code.message
        if isinstance(exception, LLMProviderError):
            return ErrorCode.LLM_NO_RESPONSE.name, ErrorCode.LLM_NO_RESPONSE.message
        if isinstance(exception, ServiceConfigurationError):
            return ErrorCode.CONFIGURATION_ERROR.name, ErrorCode.CONFIGURATION_ERROR.message
        return (
            "EVIDENCE_ORCHESTRATION_ERROR",
            f"evidence orchestration failed ({type(exception).__name__})",
        )
