from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.exceptions import AppError
from app.evaluation.experiment import load_and_validate_experiment
from app.evaluation.runner import EvaluationRunError, EvaluationRunner, normalize_contexts
from app.models.index_version import IndexVersionStatus
from app.services.index_version_service import IndexVersionService
from app.services.rag_service import RagService
from app.tasks import indexing


@pytest.mark.asyncio
async def test_version_selection_requires_accessible_status_and_tenant() -> None:
    repository = SimpleNamespace(get=AsyncMock(return_value=SimpleNamespace(status="ready")))
    service = RagService.__new__(RagService)
    service.index_version_repository = repository
    kb = SimpleNamespace(id="kb-1", active_index_version="v1")
    assert await service._resolve_index_version(None, knowledge_base=kb, tenant_id="tenant") == "v1"
    assert await service._resolve_index_version("v2", knowledge_base=kb, tenant_id="tenant") == "v2"
    repository.get.assert_awaited_once_with(tenant_id="tenant", kb_id="kb-1", version="v2")
    repository.get.return_value.status = "failed"
    with pytest.raises(AppError):
        await service._resolve_index_version("v2", knowledge_base=kb, tenant_id="tenant")


@pytest.mark.asyncio
async def test_active_version_cannot_delete_and_failed_cannot_activate() -> None:
    kb = SimpleNamespace(id="kb", active_index_version="v1")
    repository = SimpleNamespace(
        get=AsyncMock(return_value=SimpleNamespace(status=IndexVersionStatus.FAILED)),
        activate=AsyncMock(side_effect=ValueError("only ready index versions can be activated")),
    )
    service = IndexVersionService(
        session=SimpleNamespace(rollback=AsyncMock()),
        repository=repository,
        knowledge_base_repository=SimpleNamespace(get_by_id=AsyncMock(return_value=kb)),
        document_repository=SimpleNamespace(),
        indexing_service=SimpleNamespace(),
    )
    with pytest.raises(AppError):
        await service.delete(kb_id="kb", tenant_id="tenant", version="v1")
    with pytest.raises(AppError):
        await service.activate(kb_id="kb", tenant_id="tenant", version="v2")
    assert kb.active_index_version == "v1"


def test_unified_experiment_uses_business_assembled_context() -> None:
    experiment = load_and_validate_experiment(Path("eval/experiments/context_graph_upgrade.json"))
    assert experiment["changed_fields"] == ["index_version"]
    assert experiment["baseline"]["index_version"] == "v1"
    chunks = [
        {
            "chunk_id": "a",
            "content": "原文",
            "context": {
                "content": "文档：示例\n章节：一、流程\n\n父章节\n\n原文\n\n引用内容：\n补充",
                "sources": [
                    {"relation": "anchor", "content": "原文"},
                    {"relation": "reference", "content": "不应由 Runner 重新拼接"},
                ],
            },
        },
        {"chunk_id": "b", "content": "补充", "context": {"content": "补充"}},
    ]
    assert normalize_contexts(chunks, index_version="v1")[0]["content"] == "原文"
    assert normalize_contexts(chunks, index_version="v2")[0]["content"] == (
        "文档：示例\n章节：一、流程\n\n父章节\n\n原文\n\n引用内容：\n补充"
    )


def test_runner_rejects_mixed_version() -> None:
    runner = EvaluationRunner.__new__(EvaluationRunner)
    runner.experiment = {"experiment_id": "context_graph_upgrade"}
    case = {
        "id": "q",
        "question": "问",
        "ground_truth": "答",
        "case_type": "test",
        "suite": "basic",
    }
    with pytest.raises(EvaluationRunError, match="different index version"):
        runner._success_row(
            case,
            group_name="candidate",
            group={"index_version": "v2"},
            kb_alias="test",
            kb_config={"kb_id": "kb", "corpus_version": "1"},
            data={
                "metadata": {"index_version": "v2"},
                "retrieved_chunks": [{"index_version": "v1"}],
            },
            latency_ms=1.0,
            warmup=False,
        )


@pytest.mark.asyncio
async def test_v2_es_write_failure_marks_failed_and_preserves_v1(monkeypatch) -> None:
    item = SimpleNamespace(status=IndexVersionStatus.BUILDING)
    kb = SimpleNamespace(active_index_version="v1")
    document = SimpleNamespace(id="doc", status=1)
    session = SimpleNamespace(rollback=AsyncMock(), commit=AsyncMock())

    class SessionFactory:
        async def __aenter__(self):
            return session

        async def __aexit__(self, *_args):
            return None

    service = SimpleNamespace(
        purge_document_chunks=AsyncMock(),
        index_document=AsyncMock(side_effect=RuntimeError("es write failed")),
        keyword_search_provider=SimpleNamespace(close=AsyncMock()),
    )
    version_repository = SimpleNamespace(
        get=AsyncMock(return_value=item),
        mark_failed=AsyncMock(
            side_effect=lambda version, _message: setattr(version, "status", "failed")
        ),
    )
    monkeypatch.setattr(indexing, "session_factory", SessionFactory)
    monkeypatch.setattr(indexing, "build_indexing_service", lambda *_args: service)
    monkeypatch.setattr(indexing, "IndexVersionRepository", lambda _session: version_repository)
    monkeypatch.setattr(
        indexing,
        "DocumentRepository",
        lambda _session: SimpleNamespace(list_by_kb_id=AsyncMock(return_value=[document])),
    )
    monkeypatch.setattr(
        indexing,
        "KnowledgeBaseRepository",
        lambda _session: SimpleNamespace(get_by_id=AsyncMock(return_value=kb)),
    )

    with pytest.raises(RuntimeError, match="es write failed"):
        await indexing._rebuild_index_version(tenant_id="tenant", kb_id="kb", version="v2")

    assert item.status == "failed"
    assert kb.active_index_version == "v1"
    assert service.purge_document_chunks.await_count == 2
    service.purge_document_chunks.assert_any_await("doc", index_version="v2")
    version_repository.mark_failed.assert_awaited_once()
    session.rollback.assert_awaited_once()
    assert session.commit.await_count == 2  # cleaned partial data, then persisted failed status
