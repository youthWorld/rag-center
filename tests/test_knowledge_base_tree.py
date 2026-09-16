from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from app.services.knowledge_base_service import KnowledgeBaseService


class FakeKnowledgeBaseRepository:
    def __init__(self, rows):
        self.rows = rows
        self.keyword = None

    async def list_tree(self, *, tenant_id, keyword=None):
        self.tenant_id = tenant_id
        self.keyword = keyword
        return self.rows


@pytest.mark.asyncio
async def test_knowledge_base_service_groups_tree_rows_by_tenant_and_kb() -> None:
    created_at = datetime(2026, 9, 15, tzinfo=UTC)
    knowledge_base = SimpleNamespace(
        id="kb-tech",
        tenant_id="tech_position",
        name="技术岗位知识库",
        description="岗位资料",
        created_at=created_at,
    )
    rows = [
        (
            knowledge_base,
            SimpleNamespace(
                id="doc-1",
                title="backend_engineer.md",
                created_at=created_at,
            ),
            12,
        ),
        (knowledge_base, None, 0),
        (
            SimpleNamespace(
                id="kb-other",
                tenant_id="other-tenant",
                name="其他租户知识库",
                description=None,
                created_at=created_at,
            ),
            None,
            0,
        ),
    ]
    repository = FakeKnowledgeBaseRepository(rows)
    service = KnowledgeBaseService(session=SimpleNamespace(), repository=repository)

    result = await service.list_tree(tenant_id="tech_position", keyword="tech")

    assert repository.tenant_id == "tech_position"
    assert repository.keyword == "tech"
    assert [tenant.tenant_id for tenant in result] == ["tech_position"]
    assert result[0].knowledge_bases[0].kb_id == "kb-tech"
    assert result[0].knowledge_bases[0].documents[0].model_dump() == {
        "document_id": "doc-1",
        "title": "backend_engineer.md",
        "status": "SUCCESS",
        "error_message": None,
        "chunk_count": 12,
        "created_at": created_at,
    }
