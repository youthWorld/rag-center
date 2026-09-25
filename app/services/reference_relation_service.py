from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.chunk import Chunk
from app.models.chunk_relation import ChunkRelation
from app.models.document import Document, DocumentStatus
from app.utils.id_generator import generate_id


@dataclass(frozen=True, slots=True)
class ReferenceTarget:
    document_title: str | None
    heading: str | None


class ReferenceRelationService:
    """Resolve only explicit, unique references inside one KB and version."""

    _DOCUMENT_REFERENCE = re.compile(r"《(?P<title>[^》]+)》(?P<section>[^。；;，,\n]*)")
    _MARKDOWN_REFERENCE = re.compile(r"\[[^]]+\]\((?P<target>[^)#]+)(?:#(?P<section>[^)]+))?\)")
    _SECTION_REFERENCE = re.compile(r'(?:参见|见|关联规则)[：:]?\s*[“"]?(?P<section>[^”"\n]+)')

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.logger = get_logger(__name__)

    async def build_relations(
        self, *, tenant_id: str, kb_id: str, index_version: str
    ) -> dict[str, int]:
        chunks_result = await self.session.execute(
            select(Chunk, Document.title)
            .join(Document, Document.id == Chunk.document_id)
            .where(
                Chunk.tenant_id == tenant_id,
                Chunk.kb_id == kb_id,
                Chunk.index_version == index_version,
                Document.tenant_id == tenant_id,
                Document.kb_id == kb_id,
                Document.status == int(DocumentStatus.SUCCESS),
            )
            .order_by(Document.id.asc(), Chunk.order_index.asc()),
        )
        await self.session.execute(
            delete(ChunkRelation).where(
                ChunkRelation.tenant_id == tenant_id,
                ChunkRelation.kb_id == kb_id,
                ChunkRelation.index_version == index_version,
            )
        )
        rows = list(chunks_result.all())
        created = 0
        unresolved = 0
        seen_edges: set[tuple[str, str]] = set()
        for source, _ in rows:
            references = list(source.chunk_metadata.get("reference_texts", []))
            if not references:
                references = self.extract_reference_texts(source.content)
            for reference_text in references:
                target = self._parse_target(reference_text)
                candidates = self._match_target(rows, target)
                if len(candidates) != 1 or candidates[0][0].id == source.id:
                    unresolved += 1
                    self._record_unresolved(source, reference_text, len(candidates))
                    continue
                target_chunk = candidates[0][0]
                edge = (source.id, target_chunk.id)
                if edge in seen_edges:
                    continue
                seen_edges.add(edge)
                relation = ChunkRelation(
                    id=generate_id(),
                    tenant_id=tenant_id,
                    kb_id=kb_id,
                    index_version=index_version,
                    source_chunk_id=source.id,
                    target_chunk_id=target_chunk.id,
                    relation_type="reference",
                    relation_metadata={"reference_text": reference_text},
                )
                self.session.add(relation)
                created += 1
        await self.session.flush()
        return {"relation_count": created, "unresolved_count": unresolved}

    @classmethod
    def extract_reference_texts(cls, content: str) -> list[str]:
        results: list[str] = []
        for line in (content or "").splitlines():
            stripped = line.strip()
            if cls._DOCUMENT_REFERENCE.search(stripped) or cls._MARKDOWN_REFERENCE.search(stripped):
                results.append(stripped.lstrip("> "))
            elif cls._SECTION_REFERENCE.search(stripped) and any(
                marker in stripped for marker in ("参见", "见", "关联规则")
            ):
                results.append(stripped.lstrip("> "))
        return list(dict.fromkeys(results))

    @classmethod
    def _parse_target(cls, reference_text: str) -> ReferenceTarget:
        markdown = cls._MARKDOWN_REFERENCE.search(reference_text)
        if markdown:
            target = markdown.group("target") or ""
            filename = PurePosixPath(target.replace("\\", "/")).name
            return ReferenceTarget(
                document_title=re.sub(r"\.(md|markdown)$", "", filename, flags=re.I),
                heading=markdown.group("section"),
            )
        document = cls._DOCUMENT_REFERENCE.search(reference_text)
        if document:
            section = document.group("section") or None
            section = section.strip(' "“”‘’章节') if section else None
            return ReferenceTarget(document_title=document.group("title").strip(), heading=section)
        section = cls._SECTION_REFERENCE.search(reference_text)
        return ReferenceTarget(
            document_title=None, heading=section.group("section").strip() if section else None
        )

    @staticmethod
    def _match_target(
        rows: list[tuple[Chunk, str]], target: ReferenceTarget
    ) -> list[tuple[Chunk, str]]:
        candidates = rows
        if target.document_title:
            needle = re.sub(
                r"\.(md|markdown)$", "", target.document_title.strip(), flags=re.I
            ).lower()
            candidates = [
                row
                for row in candidates
                if re.sub(r"\.(md|markdown)$", "", row[1].strip(), flags=re.I).lower() == needle
            ]
        if target.heading:
            heading = target.heading.strip().lower()
            candidates = [
                row
                for row in candidates
                if str(row[0].chunk_metadata.get("heading_path") or "").lower().endswith(heading)
                or str(row[0].chunk_metadata.get("heading_path") or "").lower() == heading
            ]
        candidates = sorted(
            candidates,
            key=lambda row: (row[0].order_index is None, row[0].order_index or 0, row[0].id),
        )
        if target.document_title and not target.heading:
            document_ids = {row[0].document_id for row in candidates}
            if len(document_ids) == 1:
                return candidates[:1]
        first_by_section: dict[tuple[str, str | None], tuple[Chunk, str]] = {}
        for row in candidates:
            chunk = row[0]
            key = (chunk.document_id, chunk.section_id)
            first_by_section.setdefault(key, row)
        return list(first_by_section.values())

    @staticmethod
    def _record_unresolved(chunk: Chunk, reference_text: str, candidate_count: int) -> None:
        metadata: dict[str, Any] = dict(chunk.chunk_metadata or {})
        unresolved = list(metadata.get("unresolved_references") or [])
        item = {"reference_text": reference_text, "candidate_count": candidate_count}
        if item not in unresolved:
            unresolved.append(item)
        metadata["unresolved_references"] = unresolved
        chunk.chunk_metadata = metadata
