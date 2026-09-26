from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.repositories.chunk_repository import ChunkRepository
from app.schemas.rag import EvidenceGroup, EvidenceGroupItem, EvidenceItem, EvidencePack
from app.schemas.research import ResearchAspect, ResearchCandidate
from app.services.research_common import ResearchValidationError
from app.services.research_limits import MAX_FINAL_EVIDENCE_ITEMS


@dataclass(slots=True)
class ValidatedEvidence:
    pack: EvidencePack
    evidence_ids: dict[int, str]


def recompute_evidence_status(
    *, aspects: list[ResearchAspect], groups: list[EvidenceGroup]
) -> tuple[str, list[str]]:
    covered_by_aspect = {group.aspect: group.covered for group in groups}
    missing = [
        aspect.description
        for aspect in aspects
        if not covered_by_aspect.get(aspect.description, False)
    ]
    covered_count = len(aspects) - len(missing)
    status = (
        "complete"
        if covered_count == len(aspects)
        else "missing"
        if covered_count == 0
        else "partial"
    )
    return status, missing


def empty_evidence_pack(aspects: list[ResearchAspect]) -> EvidencePack:
    return EvidencePack(
        status="missing",
        missing_aspects=[aspect.description for aspect in aspects],
        groups=[
            EvidenceGroup(aspect=aspect.description, evidence=[], covered=False)
            for aspect in aspects
        ],
        items=[],
    )


class ResearchEvidenceService:
    def __init__(self, chunk_repository: ChunkRepository) -> None:
        self.chunk_repository = chunk_repository

    async def build_pack(
        self,
        *,
        tenant_id: str,
        index_versions: dict[str, str],
        aspects: list[ResearchAspect],
        candidates: list[ResearchCandidate],
        selections: list[list[dict[str, Any]]],
    ) -> ValidatedEvidence:
        if len(selections) != len(aspects):
            raise ResearchValidationError("evidence selections do not match aspects")
        selected_indexes: list[int] = []
        for references in selections:
            for reference in references:
                index = reference["candidate_index"]
                if index not in selected_indexes:
                    selected_indexes.append(index)
        if len(selected_indexes) > MAX_FINAL_EVIDENCE_ITEMS:
            raise ResearchValidationError("final evidence item limit exceeded")

        scopes: dict[tuple[str, str], set[str]] = {}
        for index in selected_indexes:
            candidate = candidates[index]
            if index_versions.get(candidate.kb_id) != candidate.index_version:
                raise ResearchValidationError("candidate index scope mismatch")
            scopes.setdefault((candidate.kb_id, candidate.index_version), set()).add(
                candidate.chunk_id
            )
        chunks = await self.chunk_repository.get_by_scopes(
            tenant_id=tenant_id, scopes=scopes
        )
        by_key = {
            (str(chunk.kb_id), str(chunk.index_version), str(chunk.id)): chunk
            for chunk in chunks
        }
        expected = {
            (
                candidates[index].kb_id,
                candidates[index].index_version,
                candidates[index].chunk_id,
            )
            for index in selected_indexes
        }
        if set(by_key) != expected:
            raise ResearchValidationError("selected chunks failed scope validation")

        evidence_ids: dict[int, str] = {}
        items: list[EvidenceItem] = []
        for index in selected_indexes:
            candidate = candidates[index]
            chunk = by_key[(candidate.kb_id, candidate.index_version, candidate.chunk_id)]
            evidence_id = f"E{len(items) + 1}"
            evidence_ids[index] = evidence_id
            metadata = chunk.chunk_metadata if isinstance(chunk.chunk_metadata, dict) else {}
            items.append(
                EvidenceItem(
                    evidence_id=evidence_id,
                    chunk_id=str(chunk.id),
                    kb_id=str(chunk.kb_id),
                    document_id=str(chunk.document_id),
                    title=str(chunk.title),
                    heading_path=metadata.get("heading_path"),
                    index_version=str(chunk.index_version),
                    content=str(chunk.content or ""),
                    source=candidate.source,
                    retrieved_rank=candidate.best_rank,
                    rounds=candidate.rounds,
                    query_ids=candidate.query_ids,
                )
            )

        groups: list[EvidenceGroup] = []
        for aspect, references in zip(aspects, selections, strict=True):
            group_items = [
                EvidenceGroupItem(
                    evidence_id=evidence_ids[reference["candidate_index"]],
                    role=reference["role"],
                )
                for reference in references
                if reference["candidate_index"] in evidence_ids
            ]
            groups.append(
                EvidenceGroup(
                    aspect=aspect.description,
                    evidence=group_items,
                    covered=any(item.role == "core" for item in group_items),
                )
            )
        status, missing = recompute_evidence_status(aspects=aspects, groups=groups)
        return ValidatedEvidence(
            pack=EvidencePack(
                status=status,
                missing_aspects=missing,
                groups=groups,
                items=items,
            ),
            evidence_ids=evidence_ids,
        )
