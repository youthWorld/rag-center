from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.config import Settings
from app.models.document import DocumentStatus
from app.services.document_service import DocumentService
from app.services.document_storage import DocumentStorage
from app.services.knowledge_base_service import KnowledgeBaseService


class FakeSession:
    def __init__(self) -> None:
        self.commit_count = 0
        self.rollback_count = 0

    async def commit(self) -> None:
        self.commit_count += 1

    async def rollback(self) -> None:
        self.rollback_count += 1


class FakeDocumentRepository:
    def __init__(self, document) -> None:
        self.document = document
        self.delete_by_id = AsyncMock()
        self.delete_by_kb_id = AsyncMock()

    async def get_by_id_and_tenant(self, *, document_id: str, tenant_id: str):
        if document_id == self.document.id and tenant_id == self.document.tenant_id:
            return self.document
        return None

    async def list_by_kb_id(self, *, kb_id: str, tenant_id: str):
        if kb_id == self.document.kb_id and tenant_id == self.document.tenant_id:
            return [self.document]
        return []


class FakeKnowledgeBaseRepository:
    def __init__(self, knowledge_base) -> None:
        self.knowledge_base = knowledge_base
        self.delete_by_id = AsyncMock()

    async def get_by_id_and_tenant(self, *, kb_id: str, tenant_id: str):
        if kb_id == self.knowledge_base.id and tenant_id == self.knowledge_base.tenant_id:
            return self.knowledge_base
        return None


def make_source_file(tmp_path: Path) -> Path:
    source = tmp_path / "tenant-test" / "document-test" / "sample.pdf"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"source")
    return source


def test_document_storage_rejects_unsafe_path_components(tmp_path: Path) -> None:
    storage = DocumentStorage(str(tmp_path))

    with pytest.raises(ValueError, match="tenant_id"):
        storage.path_for(
            tenant_id="../outside",
            document_id="document-test",
            filename="sample.pdf",
        )


@pytest.mark.asyncio
async def test_document_delete_removes_source_file(tmp_path: Path) -> None:
    source = make_source_file(tmp_path)
    document = SimpleNamespace(
        id="document-test",
        tenant_id="tenant-test",
        kb_id="kb-test",
        status=int(DocumentStatus.SUCCESS),
        source_file_path=str(source),
    )
    repository = FakeDocumentRepository(document)
    service = DocumentService(
        session=FakeSession(),
        document_repository=repository,
        knowledge_base_repository=SimpleNamespace(),
        indexing_service=SimpleNamespace(purge_document_chunks=AsyncMock()),
        app_settings=Settings(document_storage_path=str(tmp_path)),
    )

    await service.delete(document.id, tenant_id=document.tenant_id)

    assert not source.exists()
    assert not source.parent.exists()


@pytest.mark.asyncio
async def test_knowledge_base_delete_removes_document_source_files(tmp_path: Path) -> None:
    source = make_source_file(tmp_path)
    document = SimpleNamespace(
        id="document-test",
        tenant_id="tenant-test",
        kb_id="kb-test",
        status=int(DocumentStatus.SUCCESS),
        source_file_path=str(source),
    )
    knowledge_base = SimpleNamespace(id="kb-test", tenant_id="tenant-test")
    service = KnowledgeBaseService(
        session=FakeSession(),
        repository=FakeKnowledgeBaseRepository(knowledge_base),
        document_repository=FakeDocumentRepository(document),
        indexing_service=SimpleNamespace(purge_document_chunks=AsyncMock()),
        app_settings=Settings(document_storage_path=str(tmp_path)),
    )

    await service.delete(knowledge_base.id, tenant_id=knowledge_base.tenant_id)

    assert not source.exists()
    assert not source.parent.exists()
