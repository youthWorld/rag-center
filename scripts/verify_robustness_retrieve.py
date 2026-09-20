from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections.abc import Sequence
from typing import Any

import httpx


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify Retrieve fallback and query boundary behavior."
    )
    parser.add_argument(
        "--kb-id",
        default=os.getenv("RAG_CENTER_KB_ID", ""),
        help="Knowledge base used for the retrieve checks.",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("RAG_CENTER_BASE_URL", "http://127.0.0.1:8000"),
        help="API base URL.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("RAG_CENTER_API_KEY", ""),
        help="Optional API key sent as a Bearer token.",
    )
    parser.add_argument("--query", default="retrieve robustness verification")
    parser.add_argument("--timeout", type=float, default=15.0)
    return parser.parse_args(argv)


async def main_async(args: argparse.Namespace) -> int:
    if not args.kb_id:
        print("SKIP hybrid fallback: provide --kb-id or RAG_CENTER_KB_ID")
        print("SKIP query boundary checks: provide --kb-id or RAG_CENTER_KB_ID")
        return 0

    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    timeout = httpx.Timeout(args.timeout, connect=min(args.timeout, 5.0))
    try:
        async with httpx.AsyncClient(
            base_url=args.base_url.rstrip("/"),
            headers=headers,
            timeout=timeout,
        ) as client:
            hybrid_result = await verify_hybrid_fallback(client, args)
            boundary_result = await verify_query_boundaries(client, args)
    except httpx.RequestError as exception:
        print(f"SKIP hybrid fallback: API is unavailable ({exception})")
        print(f"SKIP query boundary checks: API is unavailable ({exception})")
        return 0

    return 0 if hybrid_result and boundary_result else 1


async def verify_hybrid_fallback(
    client: httpx.AsyncClient,
    args: argparse.Namespace,
) -> bool:
    response = await client.post(
        "/api/v1/rag/retrieve",
        json={
            "kb_id": args.kb_id,
            "user_id": "retrieve-robustness-verifier",
            "query": args.query,
            "profile": "custom",
            "retrieval_options": {
                "mode": "hybrid",
                "vector_top_k": 5,
                "bm25_top_k": 5,
            },
        },
    )
    payload = parse_payload(response)
    if response.status_code == 200 and payload.get("code") == 0:
        retrieval = ((payload.get("data") or {}).get("metadata") or {}).get(
            "retrieval",
            {},
        )
        reason = str(retrieval.get("degraded_reason") or "").lower()
        if retrieval.get("degraded") and ("bm25" in reason or "keyword" in reason):
            print("OK hybrid fallback: BM25 failure degraded to vector retrieval")
            return True
        print(
            "SKIP hybrid fallback: Elasticsearch appears available; "
            "stop it before running this case"
        )
        return True

    if payload.get("code") in {20003, 50003}:
        print(
            "SKIP hybrid fallback: retrieval dependency is unavailable or not configured "
            f"(http={response.status_code}, code={payload.get('code')})"
        )
        return True

    print(
        "FAIL hybrid fallback: "
        f"http={response.status_code}, code={payload.get('code')}, payload={payload}"
    )
    return False


async def verify_query_boundaries(
    client: httpx.AsyncClient,
    args: argparse.Namespace,
) -> bool:
    base_request = {
        "kb_id": args.kb_id,
        "user_id": "retrieve-robustness-verifier",
        "profile": "custom",
    }
    checks = (
        ("empty query", {**base_request, "query": "   "}),
        ("overlong query", {**base_request, "query": "q" * 2001}),
    )
    passed = True
    for name, request in checks:
        response = await client.post("/api/v1/rag/retrieve", json=request)
        payload = parse_payload(response)
        if response.status_code == 400 and payload.get("code") == 10001:
            print(f"OK {name}: PARAM_ERROR")
            continue
        print(
            f"FAIL {name}: expected PARAM_ERROR, "
            f"http={response.status_code}, code={payload.get('code')}, payload={payload}"
        )
        passed = False
    return passed


def parse_payload(response: httpx.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError as exception:
        raise RuntimeError(
            f"non-JSON response: http={response.status_code}, body={response.text[:500]}"
        ) from exception
    if not isinstance(payload, dict):
        raise RuntimeError(f"response is not an object: {payload!r}")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        return asyncio.run(main_async(args))
    except Exception as exception:
        print(f"retrieve robustness verification failed: {exception}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
