from __future__ import annotations

from typing import Any

from app.core.error_codes import ErrorCode
from app.core.exceptions import raise_app_error
from app.models.document import DocumentStatus
from app.tenant.plan_resolver import PlanContext


class QuotaService:
    """Check database-backed tenant quotas before mutating resources."""

    def __init__(self, knowledge_base_repository: Any, document_repository: Any) -> None:
        self.knowledge_base_repository = knowledge_base_repository
        self.document_repository = document_repository

    async def check_create_knowledge_base(
        self,
        *,
        tenant_id: str,
        plan: PlanContext,
    ) -> None:
        current = await self.knowledge_base_repository.count_by_tenant(tenant_id=tenant_id)
        limit = plan.limits.max_kb
        if current >= limit:
            raise_app_error(
                ErrorCode.QUOTA_EXCEEDED,
                f"knowledge base quota exceeded: maximum {limit} knowledge bases",
                data={"resource": "knowledge_bases", "limit": limit, "usage": current},
            )

    async def check_upload_document(
        self,
        *,
        tenant_id: str,
        kb_id: str,
        plan: PlanContext,
    ) -> None:
        document_count = await self.knowledge_base_repository.count_documents(
            kb_id=kb_id,
            tenant_id=tenant_id,
        )
        document_limit = plan.limits.max_documents_per_kb
        if document_count >= document_limit:
            raise_app_error(
                ErrorCode.QUOTA_EXCEEDED,
                f"document quota exceeded: maximum {document_limit} documents per knowledge base",
                data={
                    "resource": "documents_per_knowledge_base",
                    "limit": document_limit,
                    "usage": document_count,
                    "kb_id": kb_id,
                },
            )

        processing_count = await self.document_repository.count_by_tenant_and_status(
            tenant_id=tenant_id,
            status=int(DocumentStatus.PROCESSING),
        )
        processing_limit = plan.limits.max_processing_documents
        if processing_count >= processing_limit:
            raise_app_error(
                ErrorCode.QUOTA_EXCEEDED,
                f"indexing quota exceeded: maximum {processing_limit} processing documents",
                data={
                    "resource": "processing_documents",
                    "limit": processing_limit,
                    "usage": processing_count,
                },
            )

    async def check_reindex_document(
        self,
        *,
        tenant_id: str,
        plan: PlanContext,
    ) -> None:
        processing_count = await self.document_repository.count_by_tenant_and_status(
            tenant_id=tenant_id,
            status=int(DocumentStatus.PROCESSING),
        )
        limit = plan.limits.max_processing_documents
        if processing_count >= limit:
            raise_app_error(
                ErrorCode.QUOTA_EXCEEDED,
                f"indexing quota exceeded: maximum {limit} processing documents",
                data={
                    "resource": "processing_documents",
                    "limit": limit,
                    "usage": processing_count,
                },
            )

    async def check_knowledge_base_create(self, *, tenant_id: str, plan: PlanContext) -> None:
        await self.check_create_knowledge_base(tenant_id=tenant_id, plan=plan)

    async def check_document_upload(
        self,
        *,
        tenant_id: str,
        kb_id: str,
        plan: PlanContext,
    ) -> None:
        await self.check_upload_document(tenant_id=tenant_id, kb_id=kb_id, plan=plan)

    async def check_document_reindex(self, *, tenant_id: str, plan: PlanContext) -> None:
        await self.check_reindex_document(tenant_id=tenant_id, plan=plan)
