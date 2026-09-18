from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from scripts import reindex_knowledge_base as reindex_module


class FakeSession:
    def __init__(self) -> None:
        self.commit_count = 0
        self.rollback_count = 0

    async def commit(self) -> None:
        self.commit_count += 1

    async def rollback(self) -> None:
        self.rollback_count += 1


class FakeSessionFactory:
    def __init__(self, session: FakeSession) -> None:
        self.session = session

    def __call__(self):
        factory = self

        class SessionContext:
            async def __aenter__(self):
                return factory.session

            async def __aexit__(self, exc_type, exc, traceback):
                return False

        return SessionContext()


@pytest.mark.asyncio
async def test_reindex_one_document_purges_commits_and_queues(monkeypatch) -> None:
    session = FakeSession()
    document = SimpleNamespace(
        id="document-test",
        kb_id="kb-test",
        tenant_id="tenant-test",
        status=1,
        error_message="old error",
    )
    repository = SimpleNamespace(get_by_id=AsyncMock(return_value=document))
    monkeypatch.setattr(reindex_module, "DocumentRepository", lambda _session: repository)

    events: list[str] = []
    service = SimpleNamespace(
        keyword_search_provider=SimpleNamespace(close=AsyncMock()),
        purge_document_chunks=AsyncMock(side_effect=lambda _id: events.append("purge")),
        mark_document_failed=AsyncMock(),
    )
    task = SimpleNamespace(delay=Mock(side_effect=lambda _id: events.append("enqueue")))

    outcome = await reindex_module._reindex_one_document(
        "document-test",
        kb_id="kb-test",
        tenant_id="tenant-test",
        db_session_factory=FakeSessionFactory(session),
        task=task,
        service_builder=lambda _session, _settings: service,
        app_settings=SimpleNamespace(),
    )

    assert outcome == "queued"
    assert document.status == 3
    assert document.error_message is None
    assert events == ["purge", "enqueue"]
    assert session.commit_count == 1
    service.keyword_search_provider.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_reindex_one_document_skips_processing_document(monkeypatch) -> None:
    session = FakeSession()
    document = SimpleNamespace(
        id="document-test",
        kb_id="kb-test",
        tenant_id="tenant-test",
        status=3,
    )
    repository = SimpleNamespace(get_by_id=AsyncMock(return_value=document))
    monkeypatch.setattr(reindex_module, "DocumentRepository", lambda _session: repository)
    builder = Mock()

    outcome = await reindex_module._reindex_one_document(
        "document-test",
        kb_id="kb-test",
        tenant_id="tenant-test",
        db_session_factory=FakeSessionFactory(session),
        task=SimpleNamespace(delay=Mock()),
        service_builder=builder,
        app_settings=SimpleNamespace(),
    )

    assert outcome == "skipped"
    builder.assert_not_called()


@pytest.mark.asyncio
async def test_batch_reindex_continues_after_one_document_fails(monkeypatch, capsys) -> None:
    async def select_documents(*_args, **_kwargs):
        return ["document-1", "document-2", "document-3"], "tenant-test"

    outcomes = iter([RuntimeError("embedding failed"), "queued", "skipped"])

    async def reindex_one(*_args, **_kwargs):
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(reindex_module, "_select_document_ids", select_documents)
    monkeypatch.setattr(reindex_module, "_reindex_one_document", reindex_one)

    stats = await reindex_module.reindex_knowledge_base(
        "kb-test",
        session_factory_override=object(),
        task_override=SimpleNamespace(),
        service_builder=Mock(),
        app_settings=SimpleNamespace(),
        wait_for_completion=False,
    )

    assert stats.success == 1
    assert stats.failed == 1
    assert stats.skipped == 1
    assert "success=1 failed=1 skipped=1" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_batch_reindex_counts_worker_failure_after_waiting(monkeypatch, capsys) -> None:
    async def select_documents(*_args, **_kwargs):
        return ["document-1"], "tenant-test"

    async def reindex_one(*_args, **_kwargs):
        return "queued"

    async def wait_for_documents(*_args, **_kwargs):
        return {"document-1": (2, "embedding failed")}

    monkeypatch.setattr(reindex_module, "_select_document_ids", select_documents)
    monkeypatch.setattr(reindex_module, "_reindex_one_document", reindex_one)
    monkeypatch.setattr(reindex_module, "_wait_for_documents", wait_for_documents)

    stats = await reindex_module.reindex_knowledge_base(
        "kb-test",
        session_factory_override=object(),
        task_override=SimpleNamespace(),
        service_builder=Mock(),
        app_settings=SimpleNamespace(),
    )

    assert stats.success == 0
    assert stats.failed == 1
    assert stats.skipped == 0
    assert stats.failed_document_ids == ["document-1"]
    assert "status=2: embedding failed" in capsys.readouterr().out


def test_parse_args_supports_repeated_document_ids() -> None:
    args = reindex_module.parse_args(
        [
            "--kb-id",
            "kb-test",
            "--tenant-id",
            "tenant-test",
            "--document-id",
            "document-1",
            "--document-id",
            "document-2",
        ]
    )

    assert args.kb_id == "kb-test"
    assert args.tenant_id == "tenant-test"
    assert args.document_ids == ["document-1", "document-2"]
