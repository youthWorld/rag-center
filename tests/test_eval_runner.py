from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path
from typing import Any

import pytest

from app.evaluation.runner import EvaluationRunner
from app.evaluation.storage import read_jsonl, sanitize_snapshot

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "eval" / "datasets" / "golden_basic_20.json"
EXPERIMENT_PATH = PROJECT_ROOT / "eval" / "experiments" / "query_rewrite.json"


class FakeRetrievalClient:
    def __init__(
        self,
        *,
        delay: float = 0.0,
        fail_queries: set[str] | None = None,
        mismatch_top_k: bool = False,
    ) -> None:
        self.delay = delay
        self.fail_queries = fail_queries or set()
        self.mismatch_top_k = mismatch_top_k
        self.calls: list[dict[str, Any]] = []
        self.active = 0
        self.max_active = 0

    async def retrieve(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(copy.deepcopy(payload))
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            if payload["query"] in self.fail_queries:
                raise RuntimeError("simulated retrieve failure")
            query_options = payload["query_options"]
            rewrite_enabled = bool(query_options["enabled"])
            top_k = payload["top_k"] + int(self.mismatch_top_k)
            return {
                "query": payload["query"],
                "kb_id": payload["kb_id"],
                "retrieved_chunks": [
                    {
                        "document_id": "doc-1",
                        "chunk_id": f"chunk-{len(self.calls)}",
                        "kb_id": payload["kb_id"],
                        "title": "document.md",
                        "content": f"context for {payload['query']}",
                        "score": 0.9,
                        "retrieval_source": "vector",
                        "metadata": {"heading_path": "section"},
                    }
                ],
                "metadata": {
                    "top_k": top_k,
                    "latency_ms": 4,
                    "query_processing": {
                        "strategy": "rewrite" if rewrite_enabled else "noop",
                        "synonym_enabled": query_options["synonym_enabled"],
                        "degraded": False,
                    },
                    "retrieval": {
                        "mode": "vector",
                        "vector_top_k": 10,
                        "bm25_top_k": 0,
                        "rrf_k": None,
                    },
                    "rerank": {"enabled": False, "top_n": 5},
                    "tenant_policy": {
                        "plan": "pro",
                        "retrieve_profile": "custom",
                        "effective_mode": "vector",
                        "effective_rerank": False,
                        "effective_query_rewrite": rewrite_enabled,
                    },
                    "application_model_calls": int(rewrite_enabled),
                    "application_model_call_details": {
                        "query_rewrite": int(rewrite_enabled),
                        "rerank": 0,
                    },
                },
            }
        finally:
            self.active -= 1


def _load_inputs() -> tuple[dict, dict]:
    return (
        json.loads(EXPERIMENT_PATH.read_text(encoding="utf-8")),
        json.loads(DATASET_PATH.read_text(encoding="utf-8")),
    )


@pytest.mark.asyncio
async def test_runner_uses_bounded_concurrency_and_preserves_dataset_order(
    tmp_path: Path,
) -> None:
    experiment, dataset = _load_inputs()
    experiment["concurrency"] = 2
    client = FakeRetrievalClient(delay=0.01)
    runner = EvaluationRunner(
        project_root=PROJECT_ROOT,
        experiment=experiment,
        dataset=dataset,
        client=client,
        output_root=tmp_path,
        limit=4,
    )

    run_dir = await runner.run()

    baseline = read_jsonl(run_dir / "baseline.raw.jsonl")
    candidate = read_jsonl(run_dir / "candidate.raw.jsonl")
    expected_ids = dataset["suites"]["basic"]["case_ids"][:4]
    assert [row["case_id"] for row in baseline] == expected_ids
    assert [row["case_id"] for row in candidate] == expected_ids
    assert client.max_active == 2
    assert len(client.calls) == 8
    assert all(call["observability_enabled"] is False for call in client.calls)
    assert all(row["warmup"] is False for row in baseline + candidate)
    assert all(row["effective_config_mismatches"] == [] for row in baseline + candidate)


@pytest.mark.asyncio
async def test_explicit_warmup_is_not_written_to_formal_results(tmp_path: Path) -> None:
    experiment, dataset = _load_inputs()
    experiment["warmup_cases"] = 1
    client = FakeRetrievalClient()
    runner = EvaluationRunner(
        project_root=PROJECT_ROOT,
        experiment=experiment,
        dataset=dataset,
        client=client,
        output_root=tmp_path,
        limit=2,
    )

    run_dir = await runner.run()

    assert len(client.calls) == 6
    assert len(read_jsonl(run_dir / "baseline.raw.jsonl")) == 2
    assert len(read_jsonl(run_dir / "candidate.raw.jsonl")) == 2


@pytest.mark.asyncio
async def test_single_case_failure_is_persisted_without_cancelling_other_cases(
    tmp_path: Path,
) -> None:
    experiment, dataset = _load_inputs()
    failed_question = dataset["cases"][1]["question"]
    client = FakeRetrievalClient(fail_queries={failed_question})
    runner = EvaluationRunner(
        project_root=PROJECT_ROOT,
        experiment=experiment,
        dataset=dataset,
        client=client,
        output_root=tmp_path,
        limit=3,
    )

    run_dir = await runner.run()

    baseline = read_jsonl(run_dir / "baseline.raw.jsonl")
    candidate = read_jsonl(run_dir / "candidate.raw.jsonl")
    assert baseline[1]["error"]["message"] == "simulated retrieve failure"
    assert candidate[1]["error"]["message"] == "simulated retrieve failure"
    assert baseline[0]["error"] is None
    assert baseline[2]["error"] is None


@pytest.mark.asyncio
async def test_every_response_records_effective_config_mismatches(tmp_path: Path) -> None:
    experiment, dataset = _load_inputs()
    runner = EvaluationRunner(
        project_root=PROJECT_ROOT,
        experiment=experiment,
        dataset=dataset,
        client=FakeRetrievalClient(mismatch_top_k=True),
        output_root=tmp_path,
        limit=2,
    )

    run_dir = await runner.run()

    for row in read_jsonl(run_dir / "baseline.raw.jsonl"):
        assert row["effective_config_mismatches"] == [
            {"field": "top_k", "expected": 5, "actual": 6}
        ]


def test_snapshot_sanitizer_redacts_credentials_recursively() -> None:
    assert sanitize_snapshot(
        {
            "api_key": "secret",
            "nested": {"Authorization": "Bearer secret", "model": "demo"},
        }
    ) == {
        "api_key": "[REDACTED]",
        "nested": {"Authorization": "[REDACTED]", "model": "demo"},
    }
