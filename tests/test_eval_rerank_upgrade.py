import json
from pathlib import Path

import httpx
import pytest

from app.core.config import Settings
from app.evaluation.dataset import load_and_validate_dataset
from app.evaluation.experiment import (
    ExperimentValidationError,
    load_and_validate_experiment,
    validate_experiment,
)
from app.evaluation.metrics import score_run
from app.evaluation.report import write_report
from app.evaluation.rerank_runner import RerankEvaluationRunner
from app.evaluation.runner import EvaluationRunError, HttpRetrievalClient
from app.evaluation.storage import read_jsonl

ROOT = Path(__file__).resolve().parents[1]


class FakeClient:
    def __init__(self, experiment, *, fail=False):
        self.experiment = experiment
        self.fail = fail
        self.payloads = []

    async def retrieve(self, payload):
        self.payloads.append(payload)
        if self.fail:
            raise RuntimeError("fake retrieval failure")
        return {
            "retrieved_chunks": [
                {
                    "chunk_id": f"chunk-{i}",
                    "document_id": "doc",
                    "kb_id": payload["kb_id"],
                    "title": "title",
                    "content": f"content {i}",
                    "score": 1 / (60 + i),
                    "metadata": {"heading_path": "heading"},
                }
                for i in range(20)
            ],
            "metadata": {
                "top_k": 20,
                "latency_ms": 7,
                "application_model_calls": 0,
                "tenant_policy": {
                    "retrieve_profile": "custom",
                    "effective_mode": "hybrid",
                    "effective_rerank": False,
                    "effective_query_rewrite": False,
                    "plan": "pro",
                },
                "retrieval": {"mode": "hybrid", "vector_top_k": 20, "bm25_top_k": 20, "rrf_k": 60},
                "rerank": {"enabled": False, "top_n": 10},
                "query_processing": {"synonym_enabled": False},
            },
        }


class FakeProvider:
    def __init__(self, *, fail=False):
        self.fail = fail
        self.seen = []

    async def rerank(self, *, query, chunks, top_n):
        self.seen.append((query, [c["chunk_id"] for c in chunks], top_n))
        if self.fail:
            raise RuntimeError("rerank failure")
        return [{**c, "rerank_score": 0.5} for c in reversed(chunks)][:top_n]


class FakeScorer:
    metadata = {"provider": "fake", "model": "none"}

    def score(self, rows):
        return {row["case_id"]: {"context_recall": 1.0, "context_precision": 1.0} for row in rows}


def _runner(tmp_path, kind):
    experiment = load_and_validate_experiment(ROOT / "eval/experiments" / f"rerank_{kind}_v2.json")
    dataset = load_and_validate_dataset(ROOT / "eval/datasets/golden_basic_20.json")
    client = FakeClient(experiment)
    runner = RerankEvaluationRunner(
        project_root=ROOT,
        experiment=experiment,
        dataset=dataset,
        client=client,
        output_root=tmp_path,
        limit=2,
        settings=Settings(
            _env_file=None,
            rerank_base_url="https://workspace.cn-beijing.maas.aliyuncs.com/api/v1",
            model_api_key="fake-model-key",
            llm_api_key="fake-llm-key",
        ),
    )
    runner.providers = {"qwen37": FakeProvider(), "llm": FakeProvider()}
    return runner, client


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["effect", "upgrade"])
async def test_paired_run_freezes_one_rrf_top20_and_scores(tmp_path, kind):
    runner, client = _runner(tmp_path, kind)
    run_dir = await runner.run()
    assert len(client.payloads) == 2
    assert all(p["profile"] == "custom" and p["top_k"] == 20 for p in client.payloads)
    assert all(
        p["query_options"] == {"enabled": False, "strategy": "noop", "synonym_enabled": False}
        for p in client.payloads
    )
    assert len({p["user_id"] for p in client.payloads}) == 1
    assert client.payloads[0]["user_id"].startswith("eval-rerank-")
    shared = read_jsonl(run_dir / "shared_candidates.raw.jsonl")
    a = read_jsonl(run_dir / "baseline.raw.jsonl")
    b = read_jsonl(run_dir / "candidate.raw.jsonl")
    assert len(shared) == len(a) == len(b) == 2
    for index in range(2):
        ids = [c["chunk_id"] for c in shared[index]["candidates"]]
        assert ids == [f"chunk-{i}" for i in range(20)]
        assert a[index]["shared_candidate_hash"] == b[index]["shared_candidate_hash"]
        assert a[index]["shared_candidate_hash"] == shared[index]["candidate_hash"]
        assert len(a[index]["contexts"]) == len(b[index]["contexts"]) == 10
        assert a[index]["application_model_calls"] == int(kind == "upgrade")
        assert b[index]["application_model_calls"] == 1
        assert a[index]["effective_config_mismatches"] == []
        assert b[index]["effective_config_mismatches"] == []
        if kind == "effect":
            assert [c["chunk_id"] for c in a[index]["contexts"]] == ids[:10]
    assert runner.providers["qwen37"].seen[0][1] == ids
    if kind == "upgrade":
        assert runner.providers["llm"].seen[0][1] == ids
    else:
        assert not runner.providers["llm"].seen
    comparison = score_run(run_dir, scorer=FakeScorer())
    report = write_report(run_dir).read_text(encoding="utf-8")
    assert comparison["primary_metric"] == "context_precision"
    assert comparison["verdict"] == "preflight_only"
    assert "指标不作为正式效果结论" in report
    assert "Context Precision" in report and "Context Recall" in report
    assert ("nDCG@10" in report) == (kind == "effect")


@pytest.mark.asyncio
async def test_failure_cannot_claim_effective_result(tmp_path):
    runner, _ = _runner(tmp_path, "upgrade")
    runner.providers["qwen37"] = FakeProvider(fail=True)
    run_dir = await runner.run()
    comparison = score_run(run_dir, scorer=FakeScorer())
    assert comparison["verdict"] == "evaluation_failed"
    report = write_report(run_dir).read_text(encoding="utf-8")
    assert "不能形成启用结论" in report
    assert len(read_jsonl(run_dir / "shared_candidates.raw.jsonl")) == 2


def test_experiment_rejects_quality_profile_and_uncontrolled_changes():
    config = json.loads(
        (ROOT / "eval/experiments/rerank_effect_v2.json").read_text(encoding="utf-8")
    )
    config["baseline"]["profile"] = "quality"
    with pytest.raises(ExperimentValidationError):
        validate_experiment(config)
    config["baseline"]["profile"] = "custom"
    config["baseline"]["query_options"]["synonym_enabled"] = True
    with pytest.raises(ExperimentValidationError):
        validate_experiment(config)


@pytest.mark.asyncio
async def test_missing_workspace_url_stops_before_any_calls_or_files(tmp_path):
    runner, client = _runner(tmp_path, "effect")
    runner.settings = Settings(_env_file=None, rerank_base_url="", model_api_key="fake")
    with pytest.raises(EvaluationRunError, match="RERANK_BASE_URL"):
        await runner.run()
    assert client.payloads == []
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_access_check_rejects_wrong_tenant_or_missing_corpus():
    def handler(request):
        if request.url.path.endswith("/auth/me"):
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "data": {
                        "tenant_id": "tenant-eval",
                        "features": {
                            "hybrid_allowed": True,
                            "rerank_allowed": True,
                        },
                    },
                },
            )
        return httpx.Response(200, json={"code": 0, "data": {"document_count": 0}})

    async with HttpRetrievalClient(
        base_url="http://test",
        api_key="fake",
        timeout_seconds=5,
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(EvaluationRunError, match="different tenant"):
            await client.check_access(["kb"], "wrong-tenant")
        with pytest.raises(EvaluationRunError, match="no documents"):
            await client.check_access(["kb"], "tenant-eval")
