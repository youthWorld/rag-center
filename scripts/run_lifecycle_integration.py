from __future__ import annotations

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
from sqlalchemy import delete, func, select, update

PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
    KnowledgeBase,
    RetrievalLog,
    Tenant,
)
from app.repositories.api_key_repository import ApiKeyRepository  # noqa: E402
from app.repositories.tenant_repository import TenantRepository  # noqa: E402

BASE_URL = "http://127.0.0.1:8000"
RUN_TAG = f"lifecycle-e2e-{uuid.uuid4().hex[:12]}"
HTTP_TIMEOUT = httpx.Timeout(20.0, connect=10.0)


@dataclass(slots=True)
class TenantCredentials:
    tenant_id: str
    api_key: str
    key_hash: str


tracked_kb_ids: list[str] = []
tracked_document_ids: list[str] = []
temporary_tenants: list[TenantCredentials] = []
worker: subprocess.Popen[str] | None = None


def auth_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


def parse_response(response: httpx.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError as exc:
        raise AssertionError(
            f"non-json response: status={response.status_code} body={response.text[:500]}"
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
        f"{method.upper():6} {path:<66} "
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
    status_code: int | None = None,
) -> None:
    if status_code is None:
        assert response.status_code != 200, (response.status_code, payload)
    else:
        assert response.status_code == status_code, (response.status_code, payload)
    assert payload.get("code") == code, payload


async def create_temporary_tenant(label: str) -> TenantCredentials:
    tenant_id = f"{RUN_TAG}-{label}"
    plaintext, key_hash, key_prefix = generate_api_key()
    async with session_factory() as session:
        await TenantRepository(session).create(
            tenant_id=tenant_id,
            name=f"{RUN_TAG} {label}",
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
    temporary_tenants.append(credentials)
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
            "description": "temporary lifecycle integration test data",
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
    label: str,
) -> tuple[str, str]:
    content = (
        f"{RUN_TAG} {label}. This unique sentence verifies asynchronous indexing, "
        "retrieval, deletion, reindexing, and tenant isolation."
    )
    response, payload = await call(
        client,
        "POST",
        "/api/v1/documents/upload",
        credentials.api_key,
        json={
            "kb_id": kb_id,
            "title": f"{RUN_TAG}-{label}",
            "content": content,
        },
    )
    data = assert_success(response, payload)
    assert data["status"] == 3, data
    document_id = str(data["document_id"])
    tracked_document_ids.append(document_id)
    return document_id, content


async def wait_for_success(
    client: httpx.AsyncClient,
    credentials: TenantCredentials,
    document_id: str,
    *,
    timeout_seconds: int = 120,
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
        if data["status"] == 1:
            return data
        if data["status"] == 2:
            raise AssertionError(f"document indexing failed: {data}")
        await asyncio.sleep(1)
    raise AssertionError(f"document did not reach SUCCESS: {last_data}")


async def pg_chunk_count(document_id: str) -> int:
    async with session_factory() as session:
        result = await session.execute(
            select(func.count(Chunk.id)).where(Chunk.document_id == document_id)
        )
        return int(result.scalar_one())


async def es_chunk_count(document_id: str) -> int:
    client = AsyncElasticsearch(settings.elasticsearch_url)
    try:
        try:
            response = await client.count(
                index=settings.elasticsearch_index,
                query={"term": {"document_id": document_id}},
            )
        except NotFoundError:
            return 0
        body = getattr(response, "body", response)
        return int(body.get("count", 0))
    finally:
        await client.close()


async def force_failed(document_id: str) -> None:
    async with session_factory() as session:
        await session.execute(
            update(Document)
            .where(Document.id == document_id)
            .values(
                status=2,
                error_message="forced failure for lifecycle integration test",
            )
        )
        await session.commit()


def tree_contains_document(tree_data: Any, document_id: str) -> bool:
    if not isinstance(tree_data, list):
        return False
    return any(
        document.get("document_id") == document_id
        for tenant in tree_data
        for knowledge_base in tenant.get("knowledge_bases", [])
        for document in knowledge_base.get("documents", [])
    )


def retrieved_document_ids(payload: dict[str, Any]) -> list[str]:
    data = payload.get("data") or {}
    return [
        str(item["document_id"])
        for item in data.get("retrieved_chunks", [])
        if isinstance(item, dict) and "document_id" in item
    ]


def worker_ping() -> dict[str, Any]:
    return celery_app.control.inspect(timeout=1).ping() or {}


async def start_worker() -> None:
    global worker
    existing = await asyncio.to_thread(worker_ping)
    if existing:
        print(f"reusing existing Celery Worker: {list(existing)}", flush=True)
        return

    worker = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "celery",
            "-A",
            "app.celery_app",
            "worker",
            "--loglevel=WARNING",
            "--pool=solo",
            "--hostname=lifecycle-e2e@%h",
        ],
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if worker.poll() is not None:
            raise AssertionError(f"temporary Worker exited with code {worker.returncode}")
        if await asyncio.to_thread(worker_ping):
            print("temporary Celery Worker is ready", flush=True)
            return
        await asyncio.sleep(1)
    raise AssertionError("temporary Celery Worker did not become ready")


def stop_worker() -> None:
    global worker
    if worker is None:
        return
    if worker.poll() is None:
        worker.terminate()
        with suppress(subprocess.TimeoutExpired):
            worker.wait(timeout=15)
    if worker.poll() is None:
        worker.kill()
        worker.wait(timeout=15)
    print(f"temporary Celery Worker stopped: exit={worker.returncode}", flush=True)
    worker = None


async def cleanup() -> None:
    stop_worker()
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

    tenant_ids = [item.tenant_id for item in temporary_tenants]
    key_hashes = [item.key_hash for item in temporary_tenants]
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
        remaining_documents = await session.scalar(
            select(func.count(Document.id)).where(Document.id.in_(tracked_document_ids))
        )
        remaining_kbs = await session.scalar(
            select(func.count(KnowledgeBase.id)).where(KnowledgeBase.id.in_(tracked_kb_ids))
        )
        remaining_tenants = await session.scalar(
            select(func.count(Tenant.id)).where(Tenant.id.in_(tenant_ids))
        )
        assert int(remaining_documents or 0) == 0
        assert int(remaining_kbs or 0) == 0
        assert int(remaining_tenants or 0) == 0
    print("cleanup verified: temporary tenants, keys, documents, chunks, and ES records removed")


async def run() -> None:
    global worker
    tenant_a = await create_temporary_tenant("tenant-a")
    tenant_b = await create_temporary_tenant("tenant-b")

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=HTTP_TIMEOUT) as client:
        response, payload = await call(client, "GET", "/api/v1/auth/me", tenant_a.api_key)
        auth_data = assert_success(response, payload)
        assert auth_data["tenant_id"] == tenant_a.tenant_id, auth_data
        response, payload = await call(client, "GET", "/api/v1/auth/me", tenant_b.api_key)
        auth_data = assert_success(response, payload)
        assert auth_data["tenant_id"] == tenant_b.tenant_id, auth_data

        processing_kb = await create_kb(client, tenant_a, "processing")
        processing_document, _ = await upload_document(
            client,
            tenant_a,
            processing_kb,
            "processing-document",
        )
        response, payload = await call(
            client,
            "GET",
            f"/api/v1/documents/{processing_document}",
            tenant_a.api_key,
        )
        data = assert_success(response, payload)
        assert data["status"] == 3, data

        response, payload = await call(
            client,
            "DELETE",
            f"/api/v1/documents/{processing_document}",
            tenant_a.api_key,
        )
        assert_error(response, payload, code=10001, status_code=400)
        response, payload = await call(
            client,
            "DELETE",
            f"/api/v1/knowledge-bases/{processing_kb}",
            tenant_a.api_key,
        )
        assert_error(response, payload, code=10001, status_code=400)
        print("PASS processing document and knowledge-base delete guards")

        await start_worker()
        await wait_for_success(client, tenant_a, processing_document)

        delete_kb = await create_kb(client, tenant_a, "delete")
        delete_document, delete_content = await upload_document(
            client,
            tenant_a,
            delete_kb,
            "delete-document",
        )
        await wait_for_success(client, tenant_a, delete_document)
        assert await pg_chunk_count(delete_document) > 0
        assert await es_chunk_count(delete_document) > 0

        response, payload = await call(
            client,
            "POST",
            "/api/v1/rag/retrieve",
            tenant_a.api_key,
            json={
                "kb_id": delete_kb,
                "user_id": RUN_TAG,
                "query": delete_content,
                "top_k": 5,
            },
        )
        assert_success(response, payload)
        assert delete_document in retrieved_document_ids(payload)

        response, payload = await call(
            client,
            "DELETE",
            f"/api/v1/documents/{delete_document}",
            tenant_a.api_key,
        )
        assert_success(response, payload)
        assert await pg_chunk_count(delete_document) == 0
        assert await es_chunk_count(delete_document) == 0
        response, payload = await call(
            client,
            "GET",
            f"/api/v1/documents/{delete_document}",
            tenant_a.api_key,
        )
        assert_error(response, payload, code=10004, status_code=404)
        response, payload = await call(
            client,
            "GET",
            "/api/v1/knowledge-bases/tree",
            tenant_a.api_key,
        )
        tree_data = assert_success(response, payload)
        assert not tree_contains_document(tree_data, delete_document)
        response, payload = await call(
            client,
            "POST",
            "/api/v1/rag/retrieve",
            tenant_a.api_key,
            json={
                "kb_id": delete_kb,
                "user_id": RUN_TAG,
                "query": delete_content,
                "top_k": 5,
            },
        )
        assert_success(response, payload)
        assert delete_document not in retrieved_document_ids(payload)
        print("PASS document delete removes chunks, ES data, tree entry, and retrieval hit")

        reindex_kb = await create_kb(client, tenant_a, "reindex")
        reindex_document, _ = await upload_document(
            client,
            tenant_a,
            reindex_kb,
            "reindex-document",
        )
        await wait_for_success(client, tenant_a, reindex_document)
        assert await pg_chunk_count(reindex_document) > 0
        assert await es_chunk_count(reindex_document) > 0

        await force_failed(reindex_document)
        response, payload = await call(
            client,
            "POST",
            f"/api/v1/documents/{reindex_document}/reindex",
            tenant_a.api_key,
        )
        data = assert_success(response, payload)
        assert data["document_id"] == reindex_document
        assert data["status"] == 3
        await wait_for_success(client, tenant_a, reindex_document)
        assert await pg_chunk_count(reindex_document) > 0
        assert await es_chunk_count(reindex_document) > 0
        print("PASS FAILED document reindex reaches SUCCESS")

        stable_pg_count = await pg_chunk_count(reindex_document)
        stable_es_count = await es_chunk_count(reindex_document)
        response, payload = await call(
            client,
            "POST",
            f"/api/v1/documents/{reindex_document}/reindex",
            tenant_a.api_key,
        )
        assert_error(response, payload, code=10001, status_code=400)
        response, payload = await call(
            client,
            "GET",
            f"/api/v1/documents/{reindex_document}",
            tenant_a.api_key,
        )
        data = assert_success(response, payload)
        assert data["status"] == 1
        assert await pg_chunk_count(reindex_document) == stable_pg_count
        assert await es_chunk_count(reindex_document) == stable_es_count
        print("PASS SUCCESS document reindex is rejected without changing indexes")

        isolation_kb = await create_kb(client, tenant_a, "isolation")
        isolation_document, _ = await upload_document(
            client,
            tenant_a,
            isolation_kb,
            "isolation-document",
        )
        await wait_for_success(client, tenant_a, isolation_document)

        cross_tenant_calls = [
            ("GET", f"/api/v1/knowledge-bases/{isolation_kb}", {}),
            (
                "PATCH",
                f"/api/v1/knowledge-bases/{isolation_kb}",
                {"json": {"name": "unauthorized update"}},
            ),
            ("DELETE", f"/api/v1/knowledge-bases/{isolation_kb}", {}),
            ("GET", f"/api/v1/documents/{isolation_document}", {}),
            ("DELETE", f"/api/v1/documents/{isolation_document}", {}),
            ("POST", f"/api/v1/documents/{isolation_document}/reindex", {}),
        ]
        missing_calls = [
            ("GET", "/api/v1/knowledge-bases/missing-kb", {}),
            (
                "PATCH",
                "/api/v1/knowledge-bases/missing-kb",
                {"json": {"name": "unauthorized update"}},
            ),
            ("DELETE", "/api/v1/knowledge-bases/missing-kb", {}),
            ("GET", "/api/v1/documents/missing-document", {}),
            ("DELETE", "/api/v1/documents/missing-document", {}),
            ("POST", "/api/v1/documents/missing-document/reindex", {}),
        ]
        cross_responses = [
            await call(client, method, path, tenant_b.api_key, **kwargs)
            for method, path, kwargs in cross_tenant_calls
        ]
        missing_responses = [
            await call(client, method, path, tenant_b.api_key, **kwargs)
            for method, path, kwargs in missing_calls
        ]
        assert [
            (response.status_code, payload.get("code"), payload.get("msg"), payload.get("data"))
            for response, payload in cross_responses
        ] == [
            (response.status_code, payload.get("code"), payload.get("msg"), payload.get("data"))
            for response, payload in missing_responses
        ]
        assert all(response.status_code == 404 for response, _ in cross_responses)
        assert all(payload.get("code") == 10004 for _, payload in cross_responses)
        assert all(
            isolation_kb not in str(payload) and isolation_document not in str(payload)
            for _, payload in cross_responses
        )
        response, payload = await call(
            client,
            "GET",
            f"/api/v1/knowledge-bases/{isolation_kb}",
            tenant_a.api_key,
        )
        assert_success(response, payload)
        response, payload = await call(
            client,
            "GET",
            f"/api/v1/documents/{isolation_document}",
            tenant_a.api_key,
        )
        assert_success(response, payload)
        print("PASS cross-tenant GET/PATCH/DELETE/reindex isolation")

        for document_id in [reindex_document, isolation_document]:
            response, payload = await call(
                client,
                "DELETE",
                f"/api/v1/documents/{document_id}",
                tenant_a.api_key,
            )
            assert_success(response, payload)
        for kb_id in [processing_kb, delete_kb, reindex_kb, isolation_kb]:
            response, payload = await call(
                client,
                "DELETE",
                f"/api/v1/knowledge-bases/{kb_id}",
                tenant_a.api_key,
            )
            assert_success(response, payload)
        print("PASS API cleanup of remaining temporary resources")


async def main() -> None:
    try:
        await run()
    finally:
        if worker is None and tracked_document_ids:
            with suppress(Exception):
                await start_worker()
                await asyncio.sleep(3)
        await cleanup()


if __name__ == "__main__":
    asyncio.run(main())
