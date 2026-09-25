from __future__ import annotations

import copy
import json
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from app.core.config import Settings
from app.evaluation.evidence_runner import (
    EvidenceEvaluationRunner,
    RepositoryEvidenceVerifier,
)
from app.evaluation.storage import read_jsonl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_PATH = PROJECT_ROOT / "eval" / "experiments" / "evidence_orchestration.json"
DATASET_PATH = PROJECT_ROOT / "eval" / "datasets" / "golden_basic_20.json"


def _load_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        json.loads(EXPERIMENT_PATH.read_text(encoding="utf-8")),
        json.loads(DATASET_PATH.read_text(encoding="utf-8")),
    )


def _response(payload: dict[str, Any], *, degraded: bool = False, empty: bool = False) -> dict:
    chunks = [
        {
            "document_id": f"doc-{index}",
            "chunk_id": f"top-{index}",
            "kb_id": payload["kb_id"],
            "title": f"Top {index}",
            "content": f"raw chunk {index}",
            "context": {"content": f"assembled context {index}"},
            "score": 1 - index / 100,
            "retrieval_source": "hybrid",
            "index_version": "v2",
            "metadata": {"heading_path": f"Section {index}"},
        }
        for index in range(1, 11)
    ]
    items = (
        []
        if empty
        else [
            {
                "evidence_id": "E1",
                "chunk_id": "outside-1",
                "kb_id": payload["kb_id"],
                "document_id": "doc-outside",
                "title": "Trusted outside",
                "heading_path": "Outside section",
                "index_version": "v2",
                "content": "trusted evidence body",
                "source": "hybrid",
                "retrieved_rank": None,
            }
        ]
    )
    return {
        "query": payload["query"],
        "kb_id": payload["kb_id"],
        "retrieved_chunks": chunks,
        "evidence_pack": (
            None
            if degraded
            else {
                "status": "complete",
                "missing_aspects": [],
                "groups": [],
                "items": items,
            }
        ),
        "metadata": {
            "top_k": 10,
            "latency_ms": 100,
            "index_version": "v2",
            "query_processing": {
                "strategy": "noop",
                "synonym_enabled": False,
                "degraded": False,
            },
            "retrieval": {
                "mode": "hybrid",
                "vector_top_k": 20,
                "bm25_top_k": 20,
                "rrf_k": 10,
                "degraded": False,
            },
            "rerank": {
                "enabled": True,
                "provider": "qwen3.7",
                "top_n": 10,
                "degraded": False,
            },
            "graph_injection": {"degraded": False},
            "context_expansion": {"degraded": False},
            "tenant_policy": {
                "plan": "pro",
                "retrieve_profile": "custom",
                "effective_mode": "hybrid",
                "effective_rerank": True,
                "effective_query_rewrite": False,
                "effective_evidence": True,
            },
            "evidence": {
                "enabled": True,
                "executed": True,
                "degraded": degraded,
                "latency_ms": 25,
                "status": None if degraded else "complete",
                "model_call": {
                    "provider": "fake",
                    "request_model": "evidence-model",
                    "response_model": "evidence-model-v2",
                    "input_tokens": 40,
                    "output_tokens": 10,
                    "total_tokens": 50,
                },
            },
            "application_model_calls": 2,
            "application_model_call_details": {"rerank": 1, "evidence": 1},
        },
    }


class FakeClient:
    def __init__(self, *, degraded: bool = False, empty: bool = False) -> None:
        self.degraded = degraded
        self.empty = empty
        self.calls: list[dict[str, Any]] = []

    async def retrieve(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(copy.deepcopy(payload))
        return _response(payload, degraded=self.degraded, empty=self.empty)


class FakeVerifier:
    def __init__(self, rate: float | None = 1.0) -> None:
        self.rate = rate
        self.calls: list[dict[str, Any]] = []

    async def verify(self, *, tenant_id: str, items: list[dict[str, Any]]) -> dict[str, Any]:
        self.calls.append({"tenant_id": tenant_id, "items": items})
        verified = len(items) if self.rate == 1.0 else 0
        return {
            "total": len(items),
            "verified": verified,
            "rate": self.rate,
            "errors": [] if self.rate == 1.0 else ["mismatch"],
        }


def _runner(
    tmp_path: Path,
    *,
    client: FakeClient,
    verifier: FakeVerifier,
    limit: int = 1,
) -> EvidenceEvaluationRunner:
    experiment, dataset = _load_inputs()
    return EvidenceEvaluationRunner(
        project_root=PROJECT_ROOT,
        experiment=experiment,
        dataset=dataset,
        client=client,
        output_root=tmp_path,
        limit=limit,
        settings=Settings(_env_file=None),
        tenant_id="tenant-eval",
        verifier=verifier,
    )


@pytest.mark.asyncio
async def test_evidence_runner_uses_one_request_and_splits_contexts_latency_and_calls(
    tmp_path: Path,
) -> None:
    client = FakeClient()
    verifier = FakeVerifier()

    run_dir = await _runner(tmp_path, client=client, verifier=verifier).run()

    baseline = read_jsonl(run_dir / "baseline.raw.jsonl")[0]
    candidate = read_jsonl(run_dir / "candidate.raw.jsonl")[0]
    shared = read_jsonl(run_dir / "shared_pipeline.raw.jsonl")[0]
    assert len(client.calls) == 1
    assert client.calls[0]["evidence_options"] == {"enabled": True, "max_items": 10}
    assert baseline["contexts"][0]["content"] == "assembled context 1"
    assert candidate["contexts"] == [
        {
            "rank": 1,
            "content": "trusted evidence body",
            "document_id": "doc-outside",
            "chunk_id": "outside-1",
            "kb_id": client.calls[0]["kb_id"],
            "document": "Trusted outside",
            "heading": "Outside section",
            "retrieval_source": "hybrid",
            "index_version": "v2",
        }
    ]
    assert baseline["latency_ms"] == 100
    assert candidate["latency_ms"] == 125
    assert baseline["application_model_calls"] == 1
    assert candidate["application_model_calls"] == 2
    assert candidate["topk_outside_evidence_count"] == 1
    assert candidate["citation_traceability"]["rate"] == 1.0
    assert shared["retrieved_chunks"] == _response(client.calls[0])["retrieved_chunks"]


@pytest.mark.asyncio
async def test_evidence_degradation_preserves_baseline_and_fails_candidate(tmp_path: Path) -> None:
    run_dir = await _runner(
        tmp_path, client=FakeClient(degraded=True), verifier=FakeVerifier()
    ).run()

    baseline = read_jsonl(run_dir / "baseline.raw.jsonl")[0]
    candidate = read_jsonl(run_dir / "candidate.raw.jsonl")[0]
    assert baseline["error"] is None
    assert baseline["contexts"][0]["content"] == "assembled context 1"
    assert candidate["error"]["message"] == ("Evidence orchestration did not complete successfully")


@pytest.mark.asyncio
async def test_empty_evidence_pack_is_not_treated_as_traceable(tmp_path: Path) -> None:
    verifier = FakeVerifier()
    run_dir = await _runner(tmp_path, client=FakeClient(empty=True), verifier=verifier).run()

    candidate = read_jsonl(run_dir / "candidate.raw.jsonl")[0]
    assert candidate["error"]["message"] == "Evidence response contains no evidence items"
    assert verifier.calls == []


@pytest.mark.asyncio
async def test_traceability_failure_rejects_candidate_only(tmp_path: Path) -> None:
    run_dir = await _runner(tmp_path, client=FakeClient(), verifier=FakeVerifier(rate=0.0)).run()

    baseline = read_jsonl(run_dir / "baseline.raw.jsonl")[0]
    candidate = read_jsonl(run_dir / "candidate.raw.jsonl")[0]
    assert baseline["error"] is None
    assert candidate["error"]["message"] == ("Evidence citation traceability is below 100%")
    assert candidate["citation_traceability"]["rate"] == 0.0
    assert candidate["contexts"] == []


@pytest.mark.asyncio
async def test_repository_verifier_checks_tenant_kb_version_identity_and_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = SimpleNamespace(
        id="chunk-1",
        kb_id="kb-a",
        index_version="v2",
        document_id="doc-1",
        title="Trusted title",
        content="trusted content",
    )
    captured: dict[str, Any] = {}

    class FakeRepository:
        def __init__(self, session: object) -> None:
            captured["session"] = session

        async def get_by_scopes(self, *, tenant_id: str, scopes: dict) -> list[object]:
            captured["tenant_id"] = tenant_id
            captured["scopes"] = scopes
            return [loaded]

    @asynccontextmanager
    async def fake_session_factory():
        yield object()

    monkeypatch.setattr("app.evaluation.evidence_runner.ChunkRepository", FakeRepository)
    verifier = RepositoryEvidenceVerifier(session_factory_=fake_session_factory)
    items = [
        {
            "chunk_id": "chunk-1",
            "kb_id": "kb-a",
            "index_version": "v2",
            "document_id": "doc-1",
            "title": "Trusted title",
            "content": "tampered content",
        },
        {
            "chunk_id": "chunk-other",
            "kb_id": "kb-b",
            "index_version": "v1",
            "document_id": "doc-other",
            "title": "Other",
            "content": "other",
        },
    ]

    result = await verifier.verify(tenant_id="tenant-a", items=items)

    assert captured["tenant_id"] == "tenant-a"
    assert captured["scopes"] == {
        ("kb-a", "v2"): {"chunk-1"},
        ("kb-b", "v1"): {"chunk-other"},
    }
    assert result["rate"] == 0.0
    assert result["errors"] == ["mismatch:chunk-1", "missing:chunk-other"]
