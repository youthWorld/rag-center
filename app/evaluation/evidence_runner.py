"""Paired Evidence experiment using one real retrieval response per case."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from typing import Any, Protocol

from app.core.config import Settings
from app.db.session import session_factory
from app.evaluation.dataset import select_dataset_cases
from app.evaluation.experiment import group_request_payload
from app.evaluation.runner import (
    EvaluationRunError,
    EvaluationRunner,
    _group_manifest_summary,
    _now_iso,
)
from app.evaluation.storage import make_run_id, write_json, write_jsonl
from app.repositories.chunk_repository import ChunkRepository


class EvidenceVerifier(Protocol):
    async def verify(self, *, tenant_id: str, items: list[dict[str, Any]]) -> dict[str, Any]: ...


class RepositoryEvidenceVerifier:
    def __init__(self, session_factory_: Callable[..., Any] = session_factory) -> None:
        self.session_factory = session_factory_

    async def verify(self, *, tenant_id: str, items: list[dict[str, Any]]) -> dict[str, Any]:
        if not items:
            return {"total": 0, "verified": 0, "rate": None, "errors": ["empty pack"]}
        scopes: dict[tuple[str, str], set[str]] = {}
        for item in items:
            kb_id = str(item.get("kb_id") or "")
            version = str(item.get("index_version") or "")
            chunk_id = str(item.get("chunk_id") or "")
            if not kb_id or not version or not chunk_id:
                continue
            scopes.setdefault((kb_id, version), set()).add(chunk_id)
        async with self.session_factory() as session:
            chunks = await ChunkRepository(session).get_by_scopes(
                tenant_id=tenant_id, scopes=scopes
            )
        loaded = {(chunk.kb_id, chunk.index_version, chunk.id): chunk for chunk in chunks}
        verified = 0
        errors: list[str] = []
        for item in items:
            identity = (
                str(item.get("kb_id") or ""),
                str(item.get("index_version") or ""),
                str(item.get("chunk_id") or ""),
            )
            chunk = loaded.get(identity)
            if chunk is None:
                errors.append(f"missing:{identity[2]}")
                continue
            if (
                chunk.document_id != item.get("document_id")
                or chunk.title != item.get("title")
                or chunk.content != item.get("content")
            ):
                errors.append(f"mismatch:{identity[2]}")
                continue
            verified += 1
        total = len(items)
        return {
            "total": total,
            "verified": verified,
            "rate": verified / total if total else None,
            "errors": errors,
        }


class EvidenceEvaluationRunner(EvaluationRunner):
    def __init__(
        self,
        *,
        settings: Settings,
        tenant_id: str | None = None,
        verifier: EvidenceVerifier | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.settings = settings
        self.tenant_id = tenant_id
        self.verifier = verifier or RepositoryEvidenceVerifier()

    async def run(self):
        if not self.tenant_id:
            raise EvaluationRunError("Evidence evaluation requires --tenant-id")
        if hasattr(self.client, "check_access"):
            await self.client.check_access(
                [item["kb_id"] for item in self.dataset["knowledge_bases"].values()],
                self.tenant_id,
                require_evidence=True,
            )
        cases = select_dataset_cases(self.dataset, self.experiment["suite"])
        if self.limit is not None:
            if self.limit < 1:
                raise EvaluationRunError("limit must be positive")
            cases = cases[: self.limit]
        if not cases:
            raise EvaluationRunError("the selected suite contains no cases")

        run_id = make_run_id(self.experiment["experiment_id"], suffix=self.output_suffix)
        self.user_id = f"eval-evidence-{run_id}"[:128]
        run_dir = self.output_root / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        write_json(run_dir / "experiment.snapshot.json", self.experiment)
        write_json(run_dir / "dataset.snapshot.json", self.dataset)
        manifest = self._build_manifest(run_id=run_id, started_at=_now_iso(), cases=cases)
        manifest["models"]["rerank"] = self.settings.rerank_model
        manifest["models"]["evidence"] = self.settings.llm_model
        manifest["shared_pipeline_policy"] = (
            "one v2 hybrid Top20 + Qwen3 rerank Top10 + Evidence request per case; "
            "baseline and candidate split from the same response"
        )
        manifest["tenant_id"] = self.tenant_id
        manifest["test_user_id"] = self.user_id
        write_json(run_dir / "manifest.json", manifest)
        try:
            semaphore = asyncio.Semaphore(self.experiment["concurrency"])
            triples = await asyncio.gather(*[self._paired_case(case, semaphore) for case in cases])
            write_jsonl(run_dir / "shared_pipeline.raw.jsonl", [item[0] for item in triples])
            for name, position in (("baseline", 1), ("candidate", 2)):
                rows = [item[position] for item in triples]
                write_jsonl(run_dir / f"{name}.raw.jsonl", rows)
                manifest["groups"][name].update(_group_manifest_summary(rows))
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

    async def _paired_case(self, case: dict[str, Any], semaphore: asyncio.Semaphore):
        kb_alias = case["kb_alias"]
        kb_config = self.dataset["knowledge_bases"][kb_alias]
        payload = {
            "kb_id": kb_config["kb_id"],
            "user_id": self.user_id,
            "query": case["question"],
            **group_request_payload(self.experiment["candidate"]),
            "observability_enabled": False,
        }
        async with semaphore:
            started = time.perf_counter()
            try:
                data = await self.client.retrieve(payload)
                http_latency_ms = (time.perf_counter() - started) * 1000
                shared, baseline = self._build_shared_and_baseline(
                    case=case,
                    data=data,
                    kb_alias=kb_alias,
                    kb_config=kb_config,
                    http_latency_ms=http_latency_ms,
                )
            except Exception as exc:
                elapsed = (time.perf_counter() - started) * 1000
                shared = {"case_id": case["id"], "error": type(exc).__name__}
                rows = [
                    self._error_row(
                        case,
                        group_name=name,
                        kb_alias=kb_alias,
                        kb_config=kb_config,
                        latency_ms=elapsed,
                        exception=exc,
                        warmup=False,
                    )
                    for name in ("baseline", "candidate")
                ]
                return shared, *rows

            try:
                candidate = await self._build_candidate(
                    case=case,
                    data=data,
                    kb_alias=kb_alias,
                    kb_config=kb_config,
                    http_latency_ms=http_latency_ms,
                )
            except Exception as exc:
                candidate = self._error_row(
                    case,
                    group_name="candidate",
                    kb_alias=kb_alias,
                    kb_config=kb_config,
                    latency_ms=http_latency_ms,
                    exception=exc,
                    warmup=False,
                )
            return shared, baseline, candidate

    def _build_shared_and_baseline(
        self,
        *,
        case: dict[str, Any],
        data: dict[str, Any],
        kb_alias: str,
        kb_config: dict[str, Any],
        http_latency_ms: float,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        metadata, chunks = self._validate_shared_response(data)
        server_latency = self._number(metadata.get("latency_ms"), "metadata.latency_ms")
        total_calls = self._integer(
            metadata.get("application_model_calls"), "application_model_calls"
        )
        evidence_meta = metadata["evidence"]
        evidence_calls = int(bool(evidence_meta.get("executed")))
        baseline_data = json.loads(json.dumps(data, ensure_ascii=False))
        baseline_metadata = baseline_data["metadata"]
        baseline_metadata["tenant_policy"]["effective_evidence"] = False
        baseline_metadata["evidence"] = {
            "enabled": False,
            "executed": False,
            "degraded": False,
            "latency_ms": 0,
        }
        baseline_metadata["application_model_calls"] = max(0, total_calls - evidence_calls)
        details = dict(baseline_metadata.get("application_model_call_details") or {})
        details.pop("evidence", None)
        baseline_metadata["application_model_call_details"] = details
        baseline = self._success_row(
            case,
            group_name="baseline",
            group=self.experiment["baseline"],
            kb_alias=kb_alias,
            kb_config=kb_config,
            data=baseline_data,
            latency_ms=server_latency,
            warmup=False,
        )
        baseline["client_http_latency_ms"] = round(http_latency_ms, 3)
        shared = {
            "case_id": case["id"],
            "question": case["question"],
            "kb_id": kb_config["kb_id"],
            "retrieved_chunks": chunks,
            "evidence_pack": data.get("evidence_pack"),
            "metadata": metadata,
            "client_http_latency_ms": round(http_latency_ms, 3),
            "error": None,
        }
        return shared, baseline

    async def _build_candidate(
        self,
        *,
        case: dict[str, Any],
        data: dict[str, Any],
        kb_alias: str,
        kb_config: dict[str, Any],
        http_latency_ms: float,
    ) -> dict[str, Any]:
        metadata, chunks = self._validate_shared_response(data)
        pack = data.get("evidence_pack")
        evidence_meta = metadata["evidence"]
        if evidence_meta.get("executed") is not True or evidence_meta.get("degraded") is True:
            raise EvaluationRunError("Evidence orchestration did not complete successfully")
        if not isinstance(pack, dict):
            raise EvaluationRunError("Evidence response has no evidence_pack object")
        items = pack.get("items")
        if not isinstance(items, list) or not items:
            raise EvaluationRunError("Evidence response contains no evidence items")
        if any(not isinstance(item, dict) for item in items):
            raise EvaluationRunError("Evidence items must be objects")
        chunk_ids = [str(item.get("chunk_id") or "") for item in items]
        contents = [str(item.get("content") or "").strip() for item in items]
        if not all(chunk_ids) or len(chunk_ids) != len(set(chunk_ids)):
            raise EvaluationRunError("Evidence items contain duplicate or empty chunk ids")
        if not all(contents) or len(contents) != len(set(contents)):
            raise EvaluationRunError("Evidence items contain duplicate or empty content")
        if any(item.get("index_version") != "v2" for item in items):
            raise EvaluationRunError("Evidence items must use index v2")

        traceability = await self.verifier.verify(tenant_id=self.tenant_id or "", items=items)
        server_latency = self._number(metadata.get("latency_ms"), "metadata.latency_ms")
        evidence_latency = self._number(
            evidence_meta.get("latency_ms"), "metadata.evidence.latency_ms"
        )
        row = self._success_row(
            case,
            group_name="candidate",
            group=self.experiment["candidate"],
            kb_alias=kb_alias,
            kb_config=kb_config,
            data=data,
            latency_ms=server_latency + evidence_latency,
            warmup=False,
        )
        row["contexts"] = [
            {
                "rank": index,
                "content": item["content"],
                "document_id": item.get("document_id"),
                "chunk_id": item.get("chunk_id"),
                "kb_id": item.get("kb_id"),
                "document": item.get("title"),
                "heading": item.get("heading_path"),
                "retrieval_source": item.get("source"),
                "index_version": item.get("index_version"),
            }
            for index, item in enumerate(items, start=1)
        ]
        top_ids = {str(chunk.get("chunk_id") or "") for chunk in chunks}
        row.update(
            {
                "client_http_latency_ms": round(http_latency_ms, 3),
                "citation_traceability": traceability,
                "evidence_count": len(items),
                "evidence_chars": sum(len(item["content"]) for item in items),
                "evidence_status": pack.get("status"),
                "missing_aspects": pack.get("missing_aspects", []),
                "topk_outside_evidence_count": sum(
                    str(item.get("chunk_id") or "") not in top_ids for item in items
                ),
                "evidence_model_call": evidence_meta.get("model_call"),
            }
        )
        row["stage_metadata"]["evidence"] = evidence_meta
        if traceability.get("rate") != 1.0:
            row["contexts"] = []
            row["error"] = {
                "type": "EvaluationRunError",
                "message": "Evidence citation traceability is below 100%",
            }
        return row

    @staticmethod
    def _validate_shared_response(
        data: dict[str, Any],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        metadata = data.get("metadata")
        chunks = data.get("retrieved_chunks")
        if not isinstance(metadata, dict) or not isinstance(chunks, list):
            raise EvaluationRunError("retrieve response lacks metadata or retrieved_chunks")
        if len(chunks) != 10 or any(not isinstance(chunk, dict) for chunk in chunks):
            raise EvaluationRunError("shared pipeline must return exactly Top10 chunks")
        if metadata.get("index_version") != "v2" or any(
            chunk.get("index_version") != "v2" for chunk in chunks
        ):
            raise EvaluationRunError("shared pipeline must use index v2")
        rerank = metadata.get("rerank")
        policy = metadata.get("tenant_policy")
        evidence = metadata.get("evidence")
        if (
            not isinstance(rerank, dict)
            or rerank.get("enabled") is not True
            or rerank.get("provider") != "qwen3.7"
            or not isinstance(policy, dict)
            or policy.get("effective_query_rewrite") is not False
            or policy.get("effective_evidence") is not True
            or not isinstance(evidence, dict)
        ):
            raise EvaluationRunError("shared Evidence pipeline metadata is invalid")
        return metadata, chunks

    @staticmethod
    def _number(value: Any, field: str) -> float:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise EvaluationRunError(f"{field} must be numeric")
        return float(value)

    @staticmethod
    def _integer(value: Any, field: str) -> int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise EvaluationRunError(f"{field} must be an integer")
        return value
