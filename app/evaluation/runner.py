from __future__ import annotations

import asyncio
import os
import subprocess
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import httpx

from app.evaluation.dataset import select_dataset_cases
from app.evaluation.experiment import (
    GROUP_NAMES,
    expected_effective_config,
    group_request_payload,
)
from app.evaluation.storage import make_run_id, write_json, write_jsonl

EVAL_USER_ID = "eval_runner"


class EvaluationRunError(RuntimeError):
    pass


class RetrievalClient(Protocol):
    async def retrieve(self, payload: dict[str, Any]) -> dict[str, Any]: ...


class HttpRetrievalClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout_seconds: int,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout_seconds,
            trust_env=False,
            transport=transport,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

    async def __aenter__(self) -> HttpRetrievalClient:
        await self._client.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self._client.__aexit__(*args)

    async def retrieve(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = await self._client.post("/api/v1/rag/retrieve", json=payload)
        except httpx.HTTPError as exc:
            raise EvaluationRunError(str(exc) or type(exc).__name__) from exc
        try:
            body = response.json()
        except ValueError as exc:
            raise EvaluationRunError(
                f"retrieve returned invalid JSON (HTTP {response.status_code})"
            ) from exc
        if not 200 <= response.status_code < 300:
            raise EvaluationRunError(_response_error(body, response.status_code))
        if not isinstance(body, dict) or body.get("code", 0) != 0:
            raise EvaluationRunError(_response_error(body, response.status_code))
        data = body.get("data")
        if not isinstance(data, dict):
            raise EvaluationRunError("retrieve response has no data object")
        return data


class EvaluationRunner:
    def __init__(
        self,
        *,
        project_root: Path,
        experiment: dict[str, Any],
        dataset: dict[str, Any],
        client: RetrievalClient,
        output_root: Path,
        limit: int | None = None,
        output_suffix: str | None = None,
    ) -> None:
        self.project_root = project_root
        self.experiment = experiment
        self.dataset = dataset
        self.client = client
        self.output_root = output_root
        self.limit = limit
        self.output_suffix = output_suffix

    async def run(self) -> Path:
        cases = select_dataset_cases(self.dataset, self.experiment["suite"])
        if self.limit is not None:
            if self.limit < 1:
                raise EvaluationRunError("limit must be a positive integer")
            cases = cases[: self.limit]
        if not cases:
            raise EvaluationRunError("the selected suite contains no cases")

        run_id = make_run_id(self.experiment["experiment_id"], suffix=self.output_suffix)
        run_dir = self.output_root / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        started_at = _now_iso()
        write_json(run_dir / "experiment.snapshot.json", self.experiment)
        write_json(run_dir / "dataset.snapshot.json", self.dataset)

        manifest = self._build_manifest(run_id=run_id, started_at=started_at, cases=cases)
        write_json(run_dir / "manifest.json", manifest)
        try:
            for group_name in GROUP_NAMES:
                group = self.experiment[group_name]
                warmup_count = min(self.experiment["warmup_cases"], len(cases))
                if warmup_count:
                    warmup_rows = await self._run_group(
                        cases[:warmup_count], group_name=group_name, group=group, warmup=True
                    )
                    warmup_errors = [row for row in warmup_rows if row["error"] is not None]
                    if warmup_errors:
                        raise EvaluationRunError(
                            f"{group_name} warm-up failed for {len(warmup_errors)} cases"
                        )

                rows = await self._run_group(
                    cases, group_name=group_name, group=group, warmup=False
                )
                write_jsonl(run_dir / f"{group_name}.raw.jsonl", rows)
                manifest["groups"][group_name].update(_group_manifest_summary(rows))
                write_json(run_dir / "manifest.json", manifest)
        except Exception as exc:
            manifest["status"] = "failed"
            manifest["ended_at"] = _now_iso()
            manifest["error"] = str(exc) or type(exc).__name__
            write_json(run_dir / "manifest.json", manifest)
            raise

        manifest["status"] = "raw_complete"
        manifest["ended_at"] = _now_iso()
        write_json(run_dir / "manifest.json", manifest)
        return run_dir

    async def _run_group(
        self,
        cases: list[dict[str, Any]],
        *,
        group_name: str,
        group: dict[str, Any],
        warmup: bool,
    ) -> list[dict[str, Any]]:
        semaphore = asyncio.Semaphore(self.experiment["concurrency"])
        tasks = [
            asyncio.create_task(
                self._run_case(
                    case,
                    group_name=group_name,
                    group=group,
                    semaphore=semaphore,
                    warmup=warmup,
                )
            )
            for case in cases
        ]
        return list(await asyncio.gather(*tasks))

    async def _run_case(
        self,
        case: dict[str, Any],
        *,
        group_name: str,
        group: dict[str, Any],
        semaphore: asyncio.Semaphore,
        warmup: bool,
    ) -> dict[str, Any]:
        kb_alias = case["kb_alias"]
        kb_config = self.dataset["knowledge_bases"][kb_alias]
        payload = {
            "kb_id": kb_config["kb_id"],
            "user_id": EVAL_USER_ID,
            "query": case["question"],
            **group_request_payload(group),
            "observability_enabled": False,
        }
        async with semaphore:
            started_at = time.perf_counter()
            try:
                data = await self.client.retrieve(payload)
                latency_ms = (time.perf_counter() - started_at) * 1000
                return self._success_row(
                    case,
                    group_name=group_name,
                    group=group,
                    kb_alias=kb_alias,
                    kb_config=kb_config,
                    data=data,
                    latency_ms=latency_ms,
                    warmup=warmup,
                )
            except Exception as exc:
                latency_ms = (time.perf_counter() - started_at) * 1000
                return self._error_row(
                    case,
                    group_name=group_name,
                    kb_alias=kb_alias,
                    kb_config=kb_config,
                    latency_ms=latency_ms,
                    exception=exc,
                    warmup=warmup,
                )

    def _success_row(
        self,
        case: dict[str, Any],
        *,
        group_name: str,
        group: dict[str, Any],
        kb_alias: str,
        kb_config: dict[str, Any],
        data: dict[str, Any],
        latency_ms: float,
        warmup: bool,
    ) -> dict[str, Any]:
        metadata = data.get("metadata")
        if not isinstance(metadata, dict):
            raise EvaluationRunError("retrieve response has no metadata object")
        chunks = data.get("retrieved_chunks")
        if not isinstance(chunks, list):
            raise EvaluationRunError("retrieve response has no retrieved_chunks array")
        effective = normalize_effective_config(metadata)
        expected = expected_effective_config(group)
        mismatches = compare_effective_config(expected, effective)
        degradation = collect_degradation(metadata)
        model_calls = metadata.get("application_model_calls")
        if not isinstance(model_calls, int) or isinstance(model_calls, bool) or model_calls < 0:
            model_calls = None

        return {
            "case_id": case["id"],
            "group": group_name,
            "question": case["question"],
            "ground_truth": case["ground_truth"],
            "case_type": case["case_type"],
            "suite": case["suite"],
            "tags": case.get("tags", []),
            "kb_alias": kb_alias,
            "kb_id": kb_config["kb_id"],
            "corpus_version": kb_config["corpus_version"],
            "contexts": [
                normalize_context(chunk, rank=index)
                for index, chunk in enumerate(chunks, 1)
            ],
            "answer": None,
            "latency_ms": round(latency_ms, 3),
            "application_model_calls": model_calls,
            "application_model_call_details": metadata.get(
                "application_model_call_details"
            ),
            "effective_config": effective,
            "effective_config_expected": expected,
            "effective_config_mismatches": mismatches,
            "stage_metadata": {
                "server_latency_ms": metadata.get("latency_ms"),
                "query_processing": metadata.get("query_processing"),
                "retrieval": metadata.get("retrieval"),
                "rerank": metadata.get("rerank"),
            },
            "degradation": degradation,
            "retry": None,
            "warmup": warmup,
            "error": None,
        }

    @staticmethod
    def _error_row(
        case: dict[str, Any],
        *,
        group_name: str,
        kb_alias: str,
        kb_config: dict[str, Any],
        latency_ms: float,
        exception: Exception,
        warmup: bool,
    ) -> dict[str, Any]:
        return {
            "case_id": case["id"],
            "group": group_name,
            "question": case["question"],
            "ground_truth": case["ground_truth"],
            "case_type": case["case_type"],
            "suite": case["suite"],
            "tags": case.get("tags", []),
            "kb_alias": kb_alias,
            "kb_id": kb_config["kb_id"],
            "corpus_version": kb_config["corpus_version"],
            "contexts": [],
            "answer": None,
            "latency_ms": round(latency_ms, 3),
            "application_model_calls": None,
            "application_model_call_details": None,
            "effective_config": None,
            "effective_config_expected": None,
            "effective_config_mismatches": [],
            "stage_metadata": None,
            "degradation": [],
            "retry": None,
            "warmup": warmup,
            "error": {
                "type": type(exception).__name__,
                "message": str(exception) or type(exception).__name__,
            },
        }

    def _build_manifest(
        self, *, run_id: str, started_at: str, cases: list[dict[str, Any]]
    ) -> dict[str, Any]:
        git_commit, git_dirty = _git_state(self.project_root)
        return {
            "schema_version": "1.0",
            "run_id": run_id,
            "status": "running",
            "started_at": started_at,
            "ended_at": None,
            "git": {"commit": git_commit, "dirty": git_dirty},
            "experiment_id": self.experiment["experiment_id"],
            "dataset_version": self.dataset["version"],
            "suite": self.experiment["suite"],
            "case_count": len(cases),
            "primary_metric": self.experiment["primary_metric"],
            "corpus_versions": {
                alias: config["corpus_version"]
                for alias, config in self.dataset["knowledge_bases"].items()
            },
            "knowledge_bases": {
                alias: config["kb_id"]
                for alias, config in self.dataset["knowledge_bases"].items()
            },
            "run_config": {
                "concurrency": self.experiment["concurrency"],
                "ragas_concurrency": self.experiment["ragas_concurrency"],
                "warmup_cases": self.experiment["warmup_cases"],
                "timeout_seconds": self.experiment["timeout_seconds"],
                "limit": self.limit,
            },
            "models": {
                "application_llm": os.getenv("LLM_MODEL"),
                "embedding": os.getenv("EMBEDDING_MODEL"),
            },
            "offline_observability": {
                "ragas_do_not_track": True,
                "langfuse_request_enabled": False,
            },
            "application_model_call_counting": (
                "logical query rewrite and rerank provider invocations reported by the service; "
                "embedding and judge calls excluded"
            ),
            "groups": {
                group_name: {
                    "label": self.experiment[group_name]["label"],
                    "expected_effective_config": expected_effective_config(
                        self.experiment[group_name]
                    ),
                    "success_count": None,
                    "failure_count": None,
                }
                for group_name in GROUP_NAMES
            },
        }


def normalize_context(chunk: Any, *, rank: int) -> dict[str, Any]:
    if not isinstance(chunk, Mapping):
        return {
            "rank": rank,
            "content": "",
            "document_id": None,
            "chunk_id": None,
            "document": None,
            "heading": None,
            "score": None,
        }
    metadata = chunk.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    return {
        "rank": rank,
        "content": str(chunk.get("content") or ""),
        "document_id": chunk.get("document_id"),
        "chunk_id": chunk.get("chunk_id"),
        "kb_id": chunk.get("kb_id"),
        "document": chunk.get("title"),
        "heading": metadata.get("heading_path") or metadata.get("heading"),
        "score": chunk.get("score"),
        "vector_score": chunk.get("vector_score"),
        "bm25_score": chunk.get("bm25_score"),
        "rerank_score": chunk.get("rerank_score"),
        "retrieval_source": chunk.get("retrieval_source"),
    }


def normalize_effective_config(metadata: dict[str, Any]) -> dict[str, Any]:
    tenant_policy = metadata.get("tenant_policy")
    tenant_policy = tenant_policy if isinstance(tenant_policy, Mapping) else {}
    retrieval = metadata.get("retrieval")
    retrieval = retrieval if isinstance(retrieval, Mapping) else {}
    rerank = metadata.get("rerank")
    rerank = rerank if isinstance(rerank, Mapping) else {}
    query = metadata.get("query_processing")
    query = query if isinstance(query, Mapping) else {}
    return {
        "profile": tenant_policy.get("retrieve_profile"),
        "top_k": metadata.get("top_k"),
        "retrieval_mode": tenant_policy.get("effective_mode") or retrieval.get("mode"),
        "vector_top_k": retrieval.get("vector_top_k"),
        "bm25_top_k": retrieval.get("bm25_top_k"),
        "rrf_k": retrieval.get("rrf_k"),
        "rerank_enabled": tenant_policy.get("effective_rerank"),
        "rerank_top_n": rerank.get("top_n"),
        "rewrite_enabled": tenant_policy.get("effective_query_rewrite"),
        "synonym_enabled": query.get("synonym_enabled"),
        "plan": tenant_policy.get("plan"),
    }


def compare_effective_config(
    expected: dict[str, Any], actual: dict[str, Any]
) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if actual_value != expected_value:
            mismatches.append(
                {"field": key, "expected": expected_value, "actual": actual_value}
            )
    return mismatches


def collect_degradation(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for stage in ("query_processing", "retrieval", "rerank"):
        value = metadata.get(stage)
        if isinstance(value, Mapping) and value.get("degraded"):
            issues.append(
                {
                    "stage": stage,
                    "reason": value.get("degraded_reason") or value.get("error"),
                }
            )
    return issues


def _group_manifest_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    success_rows = [row for row in rows if row["error"] is None]
    effective_configs = []
    for row in success_rows:
        config = row.get("effective_config")
        if config is not None and config not in effective_configs:
            effective_configs.append(config)
    return {
        "success_count": len(success_rows),
        "failure_count": len(rows) - len(success_rows),
        "effective_config_mismatch_count": sum(
            bool(row.get("effective_config_mismatches")) for row in success_rows
        ),
        "degraded_case_count": sum(bool(row.get("degradation")) for row in success_rows),
        "observed_effective_configs": effective_configs,
    }


def _response_error(body: Any, status_code: int) -> str:
    if isinstance(body, dict):
        message = body.get("msg") or body.get("message")
        code = body.get("code")
        if message:
            return f"code={code}, msg={message}" if code is not None else str(message)
    return f"retrieve failed with HTTP {status_code}"


def _git_state(project_root: Path) -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return commit, dirty
    except (OSError, subprocess.SubprocessError):
        return None, None


def _now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
