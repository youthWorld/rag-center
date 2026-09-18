from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from elasticsearch import AsyncElasticsearch
from elasticsearch.exceptions import NotFoundError
from sqlalchemy import delete, select, update

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from app.celery_app import celery_app  # noqa: E402
from app.core.auth import generate_api_key  # noqa: E402
from app.core.config import settings  # noqa: E402
from app.db.session import session_factory  # noqa: E402
from app.models import (  # noqa: E402
    ApiKey,
    Chunk,
    Document,
    DocumentStatus,
    KnowledgeBase,
    RetrievalLog,
    Tenant,
)
from app.repositories.api_key_repository import ApiKeyRepository  # noqa: E402
from app.repositories.tenant_repository import TenantRepository  # noqa: E402
from app.utils.id_generator import generate_id  # noqa: E402

BASE_URL = "http://127.0.0.1:8000"
RUN_TAG = f"markdown-acceptance-{uuid.uuid4().hex[:12]}"
HTTP_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
WORKER_TIMEOUT_SECONDS = 180


@dataclass(slots=True)
class TenantCredentials:
    tenant_id: str
    api_key: str
    key_hash: str


tracked_tenants: list[TenantCredentials] = []
tracked_kb_ids: list[str] = []
tracked_document_ids: list[str] = []
manually_processing_document_ids: set[str] = set()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Markdown structured chunking and reindex acceptance test."
    )
    parser.add_argument(
        "--base-url",
        default=BASE_URL,
        help="Running API base URL.",
    )
    return parser.parse_args()


def auth_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


def parse_response(response: httpx.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError as exc:
        raise AssertionError(
            f"non-JSON response: status={response.status_code} body={response.text[:500]}"
        ) from exc
    if not isinstance(payload, dict):
        raise AssertionError(f"response is not an object: {payload!r}")
    return payload


async def call(
    client: httpx.AsyncClient,
    method: str,
    path: str,
    api_key: str,
    **kwargs: Any,
) -> tuple[httpx.Response, dict[str, Any]]:
    response = await client.request(
        method,
        path,
        headers=auth_headers(api_key),
        **kwargs,
    )
    payload = parse_response(response)
    print(
        f"{method.upper():6} {path:<72} "
        f"http={response.status_code} code={payload.get('code')}",
        flush=True,
    )
    return response, payload


def assert_success(response: httpx.Response, payload: dict[str, Any]) -> Any:
    assert response.status_code == 200, (response.status_code, payload)
    assert payload.get("code") == 0, payload
    data = payload.get("data")
    assert data is not None, payload
    return data


def assert_error(
    response: httpx.Response,
    payload: dict[str, Any],
    *,
    code: int,
    status_code: int,
) -> None:
    assert response.status_code == status_code, (response.status_code, payload)
    assert payload.get("code") == code, payload


async def create_temporary_tenant() -> TenantCredentials:
    tenant_id = f"{RUN_TAG}-tenant"
    plaintext, key_hash, key_prefix = generate_api_key()
    async with session_factory() as session:
        await TenantRepository(session).create(
            tenant_id=tenant_id,
            name=f"{RUN_TAG} tenant",
            plan="pro",
        )
        await ApiKeyRepository(session).create(
            tenant_id=tenant_id,
            key_hash=key_hash,
            key_prefix=key_prefix,
            name=f"{RUN_TAG} key",
        )
        await session.commit()
    credentials = TenantCredentials(
        tenant_id=tenant_id,
        api_key=plaintext,
        key_hash=key_hash,
    )
    tracked_tenants.append(credentials)
    return credentials


async def create_kb(
    client: httpx.AsyncClient,
    credentials: TenantCredentials,
    label: str,
) -> str:
    response, payload = await call(
        client,
        "POST",
        "/api/v1/knowledge-bases/create",
        credentials.api_key,
        json={
            "name": f"{RUN_TAG}-{label}",
            "description": "temporary Markdown reindex acceptance data",
        },
    )
    data = assert_success(response, payload)
    kb_id = str(data["kb_id"])
    tracked_kb_ids.append(kb_id)
    return kb_id


async def upload_document(
    client: httpx.AsyncClient,
    credentials: TenantCredentials,
    kb_id: str,
    title: str,
    content: str,
) -> str:
    response, payload = await call(
        client,
        "POST",
        "/api/v1/documents/upload",
        credentials.api_key,
        json={"kb_id": kb_id, "title": title, "content": content},
    )
    data = assert_success(response, payload)
    assert data["status"] == int(DocumentStatus.PROCESSING), data
    document_id = str(data["document_id"])
    tracked_document_ids.append(document_id)
    return document_id


async def wait_for_success(
    client: httpx.AsyncClient,
    credentials: TenantCredentials,
    document_id: str,
    *,
    timeout_seconds: int = WORKER_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_data: dict[str, Any] = {}
    while time.monotonic() < deadline:
        response, payload = await call(
            client,
            "GET",
            f"/api/v1/documents/{document_id}",
            credentials.api_key,
        )
        data = assert_success(response, payload)
        last_data = data
        if data["status"] == int(DocumentStatus.SUCCESS):
            return data
        if data["status"] == int(DocumentStatus.FAILED):
            raise AssertionError(f"document indexing failed: {data}")
        await asyncio.sleep(1)
    raise AssertionError(f"document did not reach SUCCESS: {last_data}")


async def fetch_chunks(document_id: str) -> list[tuple[dict[str, Any], str]]:
    async with session_factory() as session:
        result = await session.execute(
            select(Chunk.chunk_metadata, Chunk.content)
            .where(Chunk.document_id == document_id)
            .order_by(Chunk.created_at.asc(), Chunk.id.asc())
        )
        return [(dict(metadata or {}), content) for metadata, content in result.all()]


async def mutate_document_to_legacy_chunks(document_id: str) -> None:
    async with session_factory() as session:
        await session.execute(
            update(Chunk)
            .where(Chunk.document_id == document_id)
            .values(chunk_metadata={"chunk_index": 0})
        )
        await session.commit()

    es = AsyncElasticsearch(settings.elasticsearch_url)
    try:
        await es.update_by_query(
            index=settings.elasticsearch_index,
            query={"term": {"document_id": document_id}},
            script={
                "source": "ctx._source.metadata = params.metadata",
                "params": {"metadata": {"chunk_index": 0}},
            },
            conflicts="proceed",
            refresh=True,
        )
    finally:
        await es.close()


async def create_failed_worker_document(kb_id: str, tenant_id: str) -> str:
    document_id = generate_id()
    async with session_factory() as session:
        session.add(
            Document(
                id=document_id,
                tenant_id=tenant_id,
                kb_id=kb_id,
                title=f"{RUN_TAG}-worker-failure.md",
                source_type="text",
                content="   \n\n",
                status=int(DocumentStatus.SUCCESS),
            )
        )
        await session.commit()
    tracked_document_ids.append(document_id)
    return document_id


async def set_document_status(document_id: str, status: DocumentStatus) -> None:
    async with session_factory() as session:
        await session.execute(
            update(Document)
            .where(Document.id == document_id)
            .values(status=int(status), error_message=None)
        )
        await session.commit()


async def run_reindex_script(
    kb_id: str,
    *document_ids: str,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "reindex_knowledge_base.py"),
        "--kb-id",
        kb_id,
    ]
    for document_id in document_ids:
        command.extend(["--document-id", document_id])
    result = await asyncio.to_thread(
        subprocess.run,
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=WORKER_TIMEOUT_SECONDS * 2,
    )
    print("--- reindex script stdout ---")
    print(result.stdout, end="")
    if result.stderr:
        print("--- reindex script stderr ---")
        print(result.stderr, end="")
    return result


async def ensure_runtime(base_url: str) -> None:
    async with httpx.AsyncClient(base_url=base_url, timeout=HTTP_TIMEOUT) as client:
        response = await client.get("/openapi.json")
        assert response.status_code == 200, response.text[:500]

    async with session_factory() as session:
        await session.execute(select(1))

    es = AsyncElasticsearch(settings.elasticsearch_url)
    try:
        assert await es.ping(), "Elasticsearch is unavailable"
    finally:
        await es.close()

    workers = await asyncio.to_thread(
        lambda: celery_app.control.inspect(timeout=2).ping() or {}
    )
    assert workers, "Celery Worker is unavailable"
    print(f"runtime ready: API={base_url}, workers={list(workers)}", flush=True)


async def retrieve(
    client: httpx.AsyncClient,
    credentials: TenantCredentials,
    kb_id: str,
    query: str,
) -> dict[str, Any]:
    response, payload = await call(
        client,
        "POST",
        "/api/v1/rag/retrieve",
        credentials.api_key,
        json={
            "kb_id": kb_id,
            "user_id": RUN_TAG,
            "query": query,
            "profile": "balanced",
            "top_k": 5,
        },
    )
    return assert_success(response, payload)


async def verify_database_metadata(document_id: str) -> list[tuple[dict[str, Any], str]]:
    chunks = await fetch_chunks(document_id)
    assert chunks, f"document has no chunks: {document_id}"
    for metadata, content in chunks:
        assert content.strip() != "---", f"separator-only chunk found: {document_id}"
    return chunks


async def verify_elasticsearch_metadata(document_id: str) -> None:
    es = AsyncElasticsearch(settings.elasticsearch_url)
    try:
        try:
            response = await es.search(
                index=settings.elasticsearch_index,
                query={"term": {"document_id": document_id}},
                size=200,
                source=["metadata", "content"],
            )
        except NotFoundError as exc:
            raise AssertionError("Elasticsearch index does not exist") from exc
        body = getattr(response, "body", response)
        hits = body.get("hits", {}).get("hits", [])
        assert hits, f"Elasticsearch has no chunks: {document_id}"
        for hit in hits:
            source = hit.get("_source", {})
            metadata = source.get("metadata") or {}
            assert metadata.get("chunk_type"), hit
            assert metadata.get("heading_path"), hit
            assert source.get("content", "").strip() != "---", hit
    finally:
        await es.close()


async def cleanup() -> None:
    for document_id in manually_processing_document_ids:
        with suppress(Exception):
            await set_document_status(document_id, DocumentStatus.FAILED)

    deadline = time.monotonic() + WORKER_TIMEOUT_SECONDS
    while tracked_document_ids and time.monotonic() < deadline:
        async with session_factory() as session:
            result = await session.execute(
                select(Document.id, Document.status).where(
                    Document.id.in_(tracked_document_ids)
                )
            )
            pending = [
                document_id
                for document_id, status in result.all()
                if int(status) == int(DocumentStatus.PROCESSING)
            ]
        if not pending:
            break
        await asyncio.sleep(1)

    es = AsyncElasticsearch(settings.elasticsearch_url)
    try:
        for document_id in tracked_document_ids:
            with suppress(Exception):
                await es.delete_by_query(
                    index=settings.elasticsearch_index,
                    query={"term": {"document_id": document_id}},
                    conflicts="proceed",
                    refresh=True,
                )
    finally:
        await es.close()

    tenant_ids = [item.tenant_id for item in tracked_tenants]
    key_hashes = [item.key_hash for item in tracked_tenants]
    async with session_factory() as session:
        if tracked_kb_ids:
            await session.execute(
                delete(RetrievalLog).where(RetrievalLog.kb_id.in_(tracked_kb_ids))
            )
        if tracked_document_ids:
            await session.execute(
                delete(Chunk).where(Chunk.document_id.in_(tracked_document_ids))
            )
            await session.execute(
                delete(Document).where(Document.id.in_(tracked_document_ids))
            )
        if tracked_kb_ids:
            await session.execute(
                delete(KnowledgeBase).where(KnowledgeBase.id.in_(tracked_kb_ids))
            )
        if key_hashes:
            await session.execute(delete(ApiKey).where(ApiKey.key_hash.in_(key_hashes)))
        if tenant_ids:
            await session.execute(delete(Tenant).where(Tenant.id.in_(tenant_ids)))
        await session.commit()

    async with session_factory() as session:
        for model, ids in (
            (Document, tracked_document_ids),
            (KnowledgeBase, tracked_kb_ids),
            (Tenant, tenant_ids),
        ):
            if ids:
                remaining = await session.scalar(
                    select(model.id).where(model.id.in_(ids)).limit(1)
                )
                assert remaining is None, f"cleanup left {model.__tablename__}: {remaining}"
    print("cleanup verified: temporary API data, chunks, retrieval logs, and ES records removed")


async def run(base_url: str) -> None:
    await ensure_runtime(base_url)
    credentials = await create_temporary_tenant()

    heading_content = (
        f"# {RUN_TAG} 测试知识库\n\n"
        "## 第一节\n\n"
        f"{RUN_TAG} 第一节内容，用于验证 heading_path。\n\n"
        "### 1.1 子内容\n\n"
        f"{RUN_TAG} 三级标题下的内容应归入第一节。\n\n"
        "---\n\n"
        "## 第二节\n\n"
        f"{RUN_TAG} 第二节内容，应与第一节分开切块。"
    )
    table_content = (
        f"# {RUN_TAG} 表格测试\n\n"
        "## 叠加规则\n\n"
        "以下为各场景叠加情况：\n\n"
        "| 场景 | 是否可叠加 | 说明 |\n"
        "|------|-----------|------|\n"
        "| 满减 + 优惠券 | 否 | 同一订单只能选一种 |\n"
        "| 会员折扣 + 券 | 是 | 需满足会员等级 |\n"
        "| 积分 + 券 | 是 | 无额外限制 |"
    )
    legacy_content = (
        f"# {RUN_TAG} 全量 reindex 测试\n\n"
        "## 旧切块第一节\n\n"
        f"{RUN_TAG} 这篇文档会先被改写成旧 metadata，再通过批量 reindex 恢复。\n\n"
        "### 三级归属\n\n"
        f"{RUN_TAG} 三级内容属于旧切块第一节。\n\n"
        "## 旧切块第二节\n\n"
        f"{RUN_TAG} 第二节内容用于验证全量重建。"
    )
    plain_content = f"{RUN_TAG} plain text document remains indexable after reindex."

    async with httpx.AsyncClient(base_url=base_url, timeout=HTTP_TIMEOUT) as client:
        kb_id = await create_kb(client, credentials, "markdown")
        heading_document_id = await upload_document(
            client,
            credentials,
            kb_id,
            f"{RUN_TAG}-heading.md",
            heading_content,
        )
        table_document_id = await upload_document(
            client,
            credentials,
            kb_id,
            f"{RUN_TAG}-table.md",
            table_content,
        )
        legacy_document_id = await upload_document(
            client,
            credentials,
            kb_id,
            f"{RUN_TAG}-legacy.md",
            legacy_content,
        )
        plain_document_id = await upload_document(
            client,
            credentials,
            kb_id,
            f"{RUN_TAG}-plain.txt",
            plain_content,
        )

        for document_id in (
            heading_document_id,
            table_document_id,
            legacy_document_id,
            plain_document_id,
        ):
            await wait_for_success(client, credentials, document_id)

        heading_chunks = await verify_database_metadata(heading_document_id)
        heading_paths = {
            metadata.get("heading_path")
            for metadata, _content in heading_chunks
            if metadata.get("heading_path")
        }
        assert f"{RUN_TAG} 测试知识库/第一节" in heading_paths, heading_paths
        assert f"{RUN_TAG} 测试知识库/第二节" in heading_paths, heading_paths
        first_section_chunks = [
            content
            for metadata, content in heading_chunks
            if metadata.get("heading_path") == f"{RUN_TAG} 测试知识库/第一节"
        ]
        assert any("三级标题下的内容应归入第一节" in content for content in first_section_chunks)
        assert len(heading_chunks) >= 2
        print("PASS heading chunk boundaries and heading_path")

        table_chunks = await verify_database_metadata(table_document_id)
        table_pieces = [
            (metadata, content)
            for metadata, content in table_chunks
            if metadata.get("chunk_type") == "table"
        ]
        assert table_pieces, table_chunks
        assert any(
            "| 场景 | 是否可叠加 | 说明 |" in content
            or "【表头】场景 | 是否可叠加 | 说明" in content
            for _, content in table_pieces
        )
        assert any(
            "| 满减 + 优惠券 | 否 | 同一订单只能选一种 |" in content
            for _, content in table_pieces
        )
        print("PASS table chunk keeps header and complete data row")

        heading_result = await retrieve(
            client,
            credentials,
            kb_id,
            f"{RUN_TAG} 第一节内容",
        )
        heading_top = heading_result["retrieved_chunks"][0]
        assert "第一节内容" in heading_top["content"], heading_result
        assert heading_top["metadata"]["heading_path"].endswith("/第一节"), heading_top
        print("PASS heading retrieval returns first-section content and metadata")

        table_result = await retrieve(
            client,
            credentials,
            kb_id,
            "满减和优惠券能否叠加",
        )
        table_top = table_result["retrieved_chunks"][0]
        assert table_top["metadata"]["chunk_type"] == "table", table_result
        assert "满减 + 优惠券" in table_top["content"], table_result
        assert "场景" in table_top["content"], table_result
        print("PASS table retrieval returns complete table chunk and metadata")

        await verify_elasticsearch_metadata(heading_document_id)
        await verify_elasticsearch_metadata(table_document_id)
        print("PASS PostgreSQL and Elasticsearch metadata are both available")

        await mutate_document_to_legacy_chunks(legacy_document_id)
        legacy_chunks = await fetch_chunks(legacy_document_id)
        assert all("heading_path" not in metadata for metadata, _ in legacy_chunks)
        print("prepared legacy document with old chunk metadata")

        failed_worker_document_id = await create_failed_worker_document(
            kb_id,
            credentials.tenant_id,
        )

        batch_result = await run_reindex_script(kb_id)
        assert batch_result.returncode == 1, batch_result
        assert "success=4 failed=1 skipped=0" in batch_result.stdout, batch_result.stdout

        for document_id in (
            heading_document_id,
            table_document_id,
            legacy_document_id,
            plain_document_id,
        ):
            await wait_for_success(client, credentials, document_id)
        failed_response, failed_payload = await call(
            client,
            "GET",
            f"/api/v1/documents/{failed_worker_document_id}",
            credentials.api_key,
        )
        failed_data = assert_success(failed_response, failed_payload)
        assert failed_data["status"] == int(DocumentStatus.FAILED), failed_data
        assert failed_data["error_message"], failed_data

        legacy_chunks = await verify_database_metadata(legacy_document_id)
        assert any(
            metadata.get("heading_path", "").endswith("/旧切块第一节")
            for metadata, _ in legacy_chunks
        )
        assert all(metadata.get("chunk_type") for metadata, _ in legacy_chunks)
        await verify_elasticsearch_metadata(legacy_document_id)
        print(
            "PASS batch reindex waits for Worker, restores legacy metadata, "
            "and continues after failure"
        )

        skip_result = await run_reindex_script(kb_id, plain_document_id)
        assert skip_result.returncode == 0, skip_result
        assert "success=1 failed=0 skipped=0" in skip_result.stdout, skip_result.stdout

        await set_document_status(plain_document_id, DocumentStatus.PROCESSING)
        manually_processing_document_ids.add(plain_document_id)
        skip_processing_result = await run_reindex_script(kb_id, plain_document_id)
        assert skip_processing_result.returncode == 0, skip_processing_result
        assert "success=0 failed=0 skipped=1" in skip_processing_result.stdout, (
            skip_processing_result.stdout
        )
        await set_document_status(plain_document_id, DocumentStatus.SUCCESS)
        manually_processing_document_ids.discard(plain_document_id)
        print("PASS explicit document selection and PROCESSING skip")

        success_reindex_response, success_reindex_payload = await call(
            client,
            "POST",
            f"/api/v1/documents/{table_document_id}/reindex",
            credentials.api_key,
        )
        success_reindex_data = assert_success(success_reindex_response, success_reindex_payload)
        assert success_reindex_data["status"] == int(DocumentStatus.PROCESSING)
        await wait_for_success(client, credentials, table_document_id)
        await verify_elasticsearch_metadata(table_document_id)

        await set_document_status(heading_document_id, DocumentStatus.PROCESSING)
        manually_processing_document_ids.add(heading_document_id)
        processing_response, processing_payload = await call(
            client,
            "POST",
            f"/api/v1/documents/{heading_document_id}/reindex",
            credentials.api_key,
        )
        assert_error(processing_response, processing_payload, code=10001, status_code=400)
        await set_document_status(heading_document_id, DocumentStatus.SUCCESS)
        manually_processing_document_ids.discard(heading_document_id)
        print("PASS SUCCESS reindex and PROCESSING reentry rejection")

        missing_kb_result = await run_reindex_script(str(uuid.uuid4()))
        assert missing_kb_result.returncode == 2
        assert "knowledge base does not exist" in missing_kb_result.stderr
        print("PASS nonexistent knowledge base has a friendly script error")


async def main(base_url: str) -> None:
    try:
        await run(base_url)
        print("MARKDOWN REINDEX ACCEPTANCE: PASS")
    finally:
        await cleanup()


if __name__ == "__main__":
    arguments = parse_args()
    try:
        asyncio.run(main(arguments.base_url))
    except Exception as exc:
        print(f"MARKDOWN REINDEX ACCEPTANCE: FAIL: {exc}", file=sys.stderr)
        raise
