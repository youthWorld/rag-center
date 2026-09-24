"""Paired offline rerank experiment on one frozen RRF recall per case."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from typing import Any
from urllib.parse import urlparse

from app.core.config import Settings
from app.evaluation.experiment import group_request_payload
from app.evaluation.runner import (
    EvaluationRunError,
    EvaluationRunner,
    _group_manifest_summary,
    _now_iso,
)
from app.evaluation.storage import make_run_id, write_json, write_jsonl
from app.providers.llm.openai_compatible import OpenAICompatibleLLMProvider
from app.providers.rerank.qwen37 import Qwen37RerankProvider
from eval.providers.llm_rerank_baseline import LLMRerankProvider


class RerankEvaluationRunner(EvaluationRunner):
    def __init__(self, *, settings: Settings, tenant_id: str | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.settings = settings
        self.tenant_id = tenant_id
        self.providers = {"qwen37": Qwen37RerankProvider.from_settings(settings)}
        if self.experiment.get("rerank_experiment") == "upgrade":
            self.providers["llm"] = LLMRerankProvider.from_settings(
                OpenAICompatibleLLMProvider(settings),
                settings,
            )

    async def run(self):
        from app.evaluation.dataset import select_dataset_cases

        self._validate_service_configuration()
        if hasattr(self.client, "check_access"):
            await self.client.check_access(
                [item["kb_id"] for item in self.dataset["knowledge_bases"].values()],
                self.tenant_id,
            )
        cases = select_dataset_cases(self.dataset, self.experiment["suite"])
        if self.limit is not None:
            if self.limit < 1:
                raise EvaluationRunError("limit must be positive")
            cases = cases[: self.limit]
        if not cases:
            raise EvaluationRunError("the selected suite contains no cases")
        run_id = make_run_id(self.experiment["experiment_id"], suffix=self.output_suffix)
        self.user_id = f"eval-rerank-{run_id}"[:128]
        run_dir = self.output_root / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        write_json(run_dir / "experiment.snapshot.json", self.experiment)
        write_json(run_dir / "dataset.snapshot.json", self.dataset)
        manifest = self._build_manifest(run_id=run_id, started_at=_now_iso(), cases=cases)
        manifest["models"]["rerank"] = self.settings.rerank_model
        manifest["models"]["llm_baseline"] = self.settings.llm_model
        manifest["candidate_policy"] = (
            "one shared custom hybrid RRF Top20 per case; rewrite and synonyms disabled"
        )
        manifest["test_user_id"] = self.user_id
        write_json(run_dir / "manifest.json", manifest)
        try:
            semaphore = asyncio.Semaphore(self.experiment["concurrency"])
            triples = await asyncio.gather(*[self._paired_case(case, semaphore) for case in cases])
            write_jsonl(run_dir / "shared_candidates.raw.jsonl", [t[0] for t in triples])
            for name, index in (("baseline", 1), ("candidate", 2)):
                rows = [t[index] for t in triples]
                write_jsonl(run_dir / f"{name}.raw.jsonl", rows)
                manifest["groups"][name].update(_group_manifest_summary(rows))
                write_json(run_dir / "manifest.json", manifest)
        except Exception:
            manifest["status"] = "failed"
            manifest["ended_at"] = _now_iso()
            manifest["error"] = "paired execution failed; inspect raw results"
            write_json(run_dir / "manifest.json", manifest)
            raise
        manifest["status"] = "raw_complete"
        manifest["ended_at"] = _now_iso()
        write_json(run_dir / "manifest.json", manifest)
        return run_dir

    def _validate_service_configuration(self) -> None:
        url = urlparse(self.settings.rerank_base_url)
        if (
            url.scheme != "https"
            or not url.hostname
            or (
                url.hostname != "dashscope.aliyuncs.com"
                and not url.hostname.endswith(".maas.aliyuncs.com")
            )
            or "your_workspace_id" in url.hostname.lower()
            or url.path.rstrip("/") != "/api/v1"
        ):
            raise EvaluationRunError("RERANK_BASE_URL must be the DashScope HTTPS /api/v1 URL")
        if not (self.settings.rerank_api_key or self.settings.model_api_key):
            raise EvaluationRunError("RERANK_API_KEY or MODEL_API_KEY must be configured")
        if self.settings.rerank_model != "qwen3.7-text-rerank":
            raise EvaluationRunError("rerank experiment requires qwen3.7-text-rerank")
        if self.experiment["rerank_experiment"] == "upgrade" and not self.settings.llm_api_key:
            raise EvaluationRunError("LLM_API_KEY required for offline baseline")
        if not (self.settings.model_api_key or self.settings.llm_api_key):
            raise EvaluationRunError("RAGAS judge credentials are missing")

    async def _paired_case(self, case: dict[str, Any], semaphore: asyncio.Semaphore):
        kb_alias = case["kb_alias"]
        kb_config = self.dataset["knowledge_bases"][kb_alias]
        payload = {
            "kb_id": kb_config["kb_id"],
            "user_id": self.user_id,
            "query": case["question"],
            **group_request_payload(self.experiment["baseline"]),
            "observability_enabled": False,
        }
        async with semaphore:
            started = time.perf_counter()
            try:
                data = await self.client.retrieve(payload)
                elapsed = (time.perf_counter() - started) * 1000
                chunks = data.get("retrieved_chunks")
                if not isinstance(chunks, list) or len(chunks) > 20:
                    raise EvaluationRunError("shared RRF recall must contain at most 20 chunks")
                self._success_row(
                    case,
                    group_name="baseline",
                    group=self.experiment["baseline"],
                    kb_alias=kb_alias,
                    kb_config=kb_config,
                    data=data,
                    latency_ms=elapsed,
                    warmup=False,
                )
                # Snapshot the exact order/content before either provider sees it.
                frozen = json.loads(json.dumps(chunks, ensure_ascii=False))
                fingerprint = hashlib.sha256(
                    json.dumps(frozen, sort_keys=True, ensure_ascii=False).encode("utf-8")
                ).hexdigest()
                shared = {
                    "case_id": case["id"],
                    "question": case["question"],
                    "kb_id": kb_config["kb_id"],
                    "candidates": frozen,
                    "candidate_hash": fingerprint,
                    "latency_ms": round(elapsed, 3),
                    "effective_config": data["metadata"],
                    "error": None,
                }
                results = []
                for group_name in ("baseline", "candidate"):
                    results.append(
                        await self._group_from_frozen(
                            case,
                            group_name,
                            data,
                            frozen,
                            fingerprint,
                            elapsed,
                            kb_alias,
                            kb_config,
                        )
                    )
                return shared, *results
            except Exception as exc:
                elapsed = (time.perf_counter() - started) * 1000
                shared = {"case_id": case["id"], "candidates": [], "error": type(exc).__name__}
                rows = [
                    self._error_row(
                        case,
                        group_name=name,
                        kb_alias=kb_alias,
                        kb_config=kb_config,
                        latency_ms=elapsed,
                        exception=EvaluationRunError("shared recall failed"),
                        warmup=False,
                    )
                    for name in ("baseline", "candidate")
                ]
                return shared, *rows

    async def _group_from_frozen(
        self,
        case: dict[str, Any],
        group_name: str,
        data: dict[str, Any],
        frozen: list[dict[str, Any]],
        fingerprint: str,
        retrieval_ms: float,
        kb_alias: str,
        kb_config: dict[str, Any],
    ) -> dict[str, Any]:
        group = self.experiment[group_name]
        reranker = group["offline_reranker"]
        started = time.perf_counter()
        try:
            chunks = json.loads(json.dumps(frozen, ensure_ascii=False))
            if reranker == "none":
                ranked = [{**chunk, "rerank_score": None} for chunk in chunks[:10]]
            else:
                ranked = await self.providers[reranker].rerank(
                    query=case["question"],
                    chunks=chunks,
                    top_n=10,
                )
            rerank_ms = (time.perf_counter() - started) * 1000 if reranker != "none" else 0.0
            result_data = {
                **data,
                "retrieved_chunks": ranked[:10],
                "metadata": {
                    **data["metadata"],
                    "application_model_calls": int(reranker != "none"),
                },
            }
            row = self._success_row(
                case,
                group_name=group_name,
                group=group,
                kb_alias=kb_alias,
                kb_config=kb_config,
                data=result_data,
                latency_ms=retrieval_ms + rerank_ms,
                warmup=False,
            )
            row["application_model_call_details"] = {
                "query_rewrite": 0,
                "rerank": int(reranker != "none"),
            }
            row["shared_candidate_hash"] = fingerprint
            row["shared_candidate_count"] = len(frozen)
            row["stage_metadata"]["offline_rerank"] = {
                "provider": reranker,
                "model": (
                    self.settings.rerank_model
                    if reranker == "qwen37"
                    else self.settings.llm_model
                    if reranker == "llm"
                    else None
                ),
                "latency_ms": round(rerank_ms, 3),
            }
            return row
        except Exception as exc:
            row = self._error_row(
                case,
                group_name=group_name,
                kb_alias=kb_alias,
                kb_config=kb_config,
                latency_ms=retrieval_ms + (time.perf_counter() - started) * 1000,
                exception=EvaluationRunError(f"offline rerank failed ({type(exc).__name__})"),
                warmup=False,
            )
            row["shared_candidate_hash"] = fingerprint
            return row
