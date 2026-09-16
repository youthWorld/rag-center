from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import app.core.event_loop as event_loop_module
import app.db.session as db_session_module
import app.services.document_service as document_service_module
import app.tasks.indexing as indexing_task_module
from app.core.error_codes import ErrorCode
from app.core.exceptions import AppError
from app.models.document import DocumentStatus
from app.schemas.document import DocumentUploadRequest
from app.schemas.knowledge_base import KnowledgeBaseUpdateRequest
from app.services.document_service import DocumentService
from app.services.indexing_service import IndexingService
from app.services.knowledge_base_service import KnowledgeBaseService
from app.tasks.indexing import index_document_task
from app.utils.text_splitter import TextSplitter


class FakeSession:
    def __init__(self) -> None:
        self.commit_count = 0
        self.rollback_count = 0

    async def commit(self) -> None:
        self.commit_count += 1

    async def rollback(self) -> None:
        self.rollback_count += 1

    async def refresh(self, _record) -> None:
        return None


class FakeDocumentRepository:
    def __init__(self, document=None, *, chunk_count: int = 0) -> None:
        self.document = document
        self.chunk_count = chunk_count
        self.deleted = []

    async def get_by_id_and_tenant(self, *, document_id: str, tenant_id: str):
        if (
            self.document
            and self.document.id == document_id
            and self.document.tenant_id == tenant_id
        ):
            return self.document
        return None

    async def get_by_id(self, document_id: str, tenant_id=None):
        del tenant_id
        return self.document if self.document and self.document.id == document_id else None

    async def count_chunks(self, *, document_id: str) -> int:
        assert document_id == self.document.id
        return self.chunk_count

    async def list_by_kb_id(self, *, kb_id: str, tenant_id: str):
        if self.document and self.document.kb_id == kb_id and self.document.tenant_id == tenant_id:
            return [self.document]
        return []

    async def delete_by_id(self, **kwargs) -> None:
        self.deleted.append(("document", kwargs))

    async def delete_by_kb_id(self, **kwargs) -> None:
        self.deleted.append(("kb_documents", kwargs))


class FakeKnowledgeBaseRepository:
    def __init__(self, knowledge_base) -> None:
        self.knowledge_base = knowledge_base
        self.deleted = []

    async def get_by_id_and_tenant(self, *, kb_id: str, tenant_id: str):
        if self.knowledge_base.id == kb_id and self.knowledge_base.tenant_id == tenant_id:
            return self.knowledge_base
        return None

    async def count_documents(self, *, kb_id: str, tenant_id: str) -> int:
        assert kb_id == self.knowledge_base.id
        assert tenant_id == self.knowledge_base.tenant_id
        return 1

    async def delete_by_id(self, **kwargs) -> None:
        self.deleted.append(("kb", kwargs))


def make_document(*, status: int = int(DocumentStatus.FAILED)):
    return SimpleNamespace(
        id="document-test",
        tenant_id="tenant-test",
        kb_id="kb-test",
        title="Test",
        content="first paragraph\n\nsecond paragraph",
        source_type="text",
        status=status,
        error_message="previous error" if status == int(DocumentStatus.FAILED) else None,
        created_at=datetime(2026, 9, 16, tzinfo=UTC),
        updated_at=datetime(2026, 9, 16, tzinfo=UTC),
    )


@pytest.mark.asyncio
async def test_upload_creates_processing_record_and_enqueues_task(monkeypatch) -> None:
    session = FakeSession()
    document = make_document(status=int(DocumentStatus.PROCESSING))
    indexing = SimpleNamespace(create_document_record=AsyncMock(return_value=document))
    service = DocumentService(
        session=session,
        document_repository=FakeDocumentRepository(document),
        knowledge_base_repository=SimpleNamespace(),
        indexing_service=indexing,
    )
    delay = Mock()
    monkeypatch.setattr(
        document_service_module,
        "index_document_task",
        SimpleNamespace(delay=delay),
    )

    result = await service.upload(
        DocumentUploadRequest(kb_id="kb-test", title="Test", content="content"),
        tenant_id="tenant-test",
    )

    assert result.document_id == "document-test"
    assert result.status == int(DocumentStatus.PROCESSING)
    assert result.chunk_count == 0
    delay.assert_called_once_with("document-test")


@pytest.mark.asyncio
async def test_reindex_purges_old_chunks_before_enqueuing(monkeypatch) -> None:
    session = FakeSession()
    document = make_document(status=int(DocumentStatus.FAILED))
    events: list[str] = []
    indexing = SimpleNamespace(
        purge_document_chunks=AsyncMock(side_effect=lambda _id: events.append("purge")),
        mark_document_failed=AsyncMock(),
    )
    service = DocumentService(
        session=session,
        document_repository=FakeDocumentRepository(document),
        knowledge_base_repository=SimpleNamespace(),
        indexing_service=indexing,
    )
    delay = Mock(side_effect=lambda _id: events.append("enqueue"))
    monkeypatch.setattr(
        document_service_module,
        "index_document_task",
        SimpleNamespace(delay=delay),
    )

    result = await service.reindex("document-test", tenant_id="tenant-test")

    assert result.status == int(DocumentStatus.PROCESSING)
    assert document.status == int(DocumentStatus.PROCESSING)
    assert document.error_message is None
    assert events == ["purge", "enqueue"]
    assert session.commit_count == 1


@pytest.mark.asyncio
async def test_delete_rejects_processing_document() -> None:
    document = make_document(status=int(DocumentStatus.PROCESSING))
    service = DocumentService(
        session=FakeSession(),
        document_repository=FakeDocumentRepository(document),
        knowledge_base_repository=SimpleNamespace(),
        indexing_service=SimpleNamespace(),
    )

    with pytest.raises(AppError) as raised:
        await service.delete("document-test", tenant_id="tenant-test")

    assert raised.value.code == ErrorCode.PARAM_ERROR.code


@pytest.mark.asyncio
async def test_knowledge_base_update_replaces_settings_and_validates_synonyms() -> None:
    knowledge_base = SimpleNamespace(
        id="kb-test",
        tenant_id="tenant-test",
        name="Old name",
        description="Old description",
        settings={"rewrite_hint": "old"},
        created_at=datetime(2026, 9, 16, tzinfo=UTC),
        updated_at=datetime(2026, 9, 16, tzinfo=UTC),
    )
    repository = FakeKnowledgeBaseRepository(knowledge_base)
    service = KnowledgeBaseService(session=FakeSession(), repository=repository)

    result = await service.update(
        "kb-test",
        KnowledgeBaseUpdateRequest(
            name="New name",
            settings={"synonyms": [{"terms": ["raw"], "expand": ["expanded"]}]},
        ),
        tenant_id="tenant-test",
    )

    assert result.name == "New name"
    assert result.settings == {
        "synonyms": [{"terms": ["raw"], "expand": ["expanded"]}]
    }

    with pytest.raises(AppError) as raised:
        await service.update(
            "kb-test",
            KnowledgeBaseUpdateRequest(
                settings={"synonyms": [{"terms": [], "expand": ["expanded"]}]}
            ),
            tenant_id="tenant-test",
        )
    assert raised.value.code == ErrorCode.PARAM_ERROR.code


@pytest.mark.asyncio
async def test_knowledge_base_delete_purges_before_database_rows() -> None:
    knowledge_base = SimpleNamespace(
        id="kb-test",
        tenant_id="tenant-test",
        name="Test",
        description=None,
    )
    document = make_document(status=int(DocumentStatus.SUCCESS))
    document_repository = FakeDocumentRepository(document)
    knowledge_repository = FakeKnowledgeBaseRepository(knowledge_base)
    events: list[str] = []
    indexing = SimpleNamespace(
        purge_document_chunks=AsyncMock(side_effect=lambda _id: events.append("purge"))
    )
    document_repository.delete_by_kb_id = AsyncMock(
        side_effect=lambda **_kwargs: events.append("documents")
    )
    knowledge_repository.delete_by_id = AsyncMock(side_effect=lambda **_kwargs: events.append("kb"))
    service = KnowledgeBaseService(
        session=FakeSession(),
        repository=knowledge_repository,
        document_repository=document_repository,
        indexing_service=indexing,
    )

    await service.delete("kb-test", tenant_id="tenant-test")

    assert events == ["purge", "documents", "kb"]


@pytest.mark.asyncio
async def test_indexing_service_updates_status_after_background_index() -> None:
    document = make_document(status=int(DocumentStatus.PROCESSING))
    session = FakeSession()

    class DocumentRepositoryStub:
        async def get_by_id(self, document_id: str, tenant_id=None):
            del tenant_id
            return document if document_id == document.id else None

    class KnowledgeBaseRepositoryStub:
        pass

    class EmbeddingStub:
        async def embed_documents(self, texts):
            return [[float(index)] for index, _ in enumerate(texts)]

    class VectorStoreStub:
        def __init__(self):
            self.chunks = []

        async def add_chunks(self, chunks):
            self.chunks.extend(chunks)

        async def delete_by_document_id(self, _document_id):
            return None

    service = IndexingService(
        session=session,
        document_repository=DocumentRepositoryStub(),
        knowledge_base_repository=KnowledgeBaseRepositoryStub(),
        splitter=TextSplitter(chunk_size=100, chunk_overlap=10),
        embedding_provider=EmbeddingStub(),
        vector_store=VectorStoreStub(),
    )

    count = await service.index_existing_document(document.id)

    assert count == 1
    assert document.status == int(DocumentStatus.SUCCESS)
    assert document.error_message is None
    assert session.commit_count == 1


@pytest.mark.asyncio
async def test_indexing_service_marks_document_failed_when_loading_document_fails() -> None:
    document = make_document(status=int(DocumentStatus.PROCESSING))
    session = FakeSession()

    class FlakyDocumentRepository:
        def __init__(self) -> None:
            self.calls = 0

        async def get_by_id(self, document_id: str, tenant_id=None):
            del tenant_id
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("document lookup failed")
            return document if document_id == document.id else None

    class KnowledgeBaseRepositoryStub:
        pass

    service = IndexingService(
        session=session,
        document_repository=FlakyDocumentRepository(),
        knowledge_base_repository=KnowledgeBaseRepositoryStub(),
        splitter=TextSplitter(chunk_size=100, chunk_overlap=10),
        embedding_provider=Mock(),
        vector_store=Mock(),
    )

    with pytest.raises(RuntimeError, match="document lookup failed"):
        await service.index_existing_document(document.id)

    assert document.status == int(DocumentStatus.FAILED)
    assert document.error_message == "document lookup failed"
    assert session.rollback_count == 1
    assert session.commit_count == 1


def test_worker_async_runner_uses_selector_event_loop_on_windows(monkeypatch) -> None:
    selector_policy = object()
    create_policy = Mock(return_value=selector_policy)
    set_policy = Mock()
    monkeypatch.setattr(indexing_task_module.sys, "platform", "win32")
    monkeypatch.setattr(
        indexing_task_module.asyncio,
        "WindowsSelectorEventLoopPolicy",
        create_policy,
        raising=False,
    )
    monkeypatch.setattr(indexing_task_module.asyncio, "set_event_loop_policy", set_policy)

    async def operation() -> str:
        return "done"

    assert indexing_task_module._run_async(operation()) == "done"
    create_policy.assert_called_once_with()
    set_policy.assert_called_once_with(selector_policy)


def test_database_session_configures_selector_event_loop_on_windows(monkeypatch) -> None:
    selector_policy = object()
    create_policy = Mock(return_value=selector_policy)
    set_policy = Mock()
    monkeypatch.setattr(db_session_module.sys, "platform", "win32")
    monkeypatch.setattr(
        db_session_module.asyncio,
        "WindowsSelectorEventLoopPolicy",
        create_policy,
        raising=False,
    )
    monkeypatch.setattr(db_session_module.asyncio, "set_event_loop_policy", set_policy)

    db_session_module._configure_asyncio_policy()

    create_policy.assert_called_once_with()
    set_policy.assert_called_once_with(selector_policy)


def test_uvicorn_loop_factory_uses_selector_event_loop_on_windows(monkeypatch) -> None:
    monkeypatch.setattr(event_loop_module.sys, "platform", "win32")

    loop = event_loop_module.selector_event_loop_factory(use_subprocess=True)

    try:
        assert isinstance(loop, event_loop_module.asyncio.SelectorEventLoop)
    finally:
        loop.close()


def test_celery_task_uses_requested_retry_policy() -> None:
    assert index_document_task.max_retries == 2
    assert index_document_task.default_retry_delay == 10
