from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from app.evaluation.metrics import (
    build_comparison,
    build_group_metrics,
    linear_percentile,
    score_run,
)
from app.evaluation.storage import write_json, write_jsonl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_PATH = PROJECT_ROOT / "eval" / "experiments" / "query_rewrite.json"
EVIDENCE_EXPERIMENT = PROJECT_ROOT / "eval" / "experiments" / "evidence_orchestration.json"
DATASET_PATH = PROJECT_ROOT / "eval" / "datasets" / "golden_basic_20.json"


class StaticScorer:
    metadata = {
        "model": "judge-test",
        "temperature": 0.0,
        "timeout_seconds": 120,
        "max_workers": 4,
    }

    def __init__(self, recall: float, precision: float) -> None:
        self.recall = recall
        self.precision = precision
        self.calls = 0

    def score(self, rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        self.calls += 1
        return {
            row["case_id"]: {
                "context_recall": self.recall,
                "context_precision": self.precision,
                "error": None,
            }
            for row in rows
        }


def _row(
    case_id: str,
    *,
    latency: float | None = 100.0,
    calls: int | None = 0,
    error: dict | None = None,
    warmup: bool = False,
    mismatch: bool = False,
    degraded: bool = False,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "question": f"question {case_id}",
        "ground_truth": "answer",
        "case_type": "direct_fact",
        "tags": ["tag"],
        "contexts": [{"content": "context"}],
        "latency_ms": latency,
        "application_model_calls": calls,
        "effective_config_mismatches": ([{"field": "top_k"}] if mismatch else []),
        "degradation": ([{"stage": "retrieval"}] if degraded else []),
        "warmup": warmup,
        "error": error,
    }


def _group_metrics(
    recall: float,
    precision: float,
    p95: float,
    calls: float,
    *,
    failed: int = 0,
    quality_count: int = 20,
    call_count: int = 20,
    mismatches: int = 0,
    degraded: int = 0,
) -> dict[str, Any]:
    successful = 20 - failed
    return {
        "case_count": 20,
        "successful_cases": successful,
        "failed_cases": failed,
        "quality": {
            "context_recall": recall,
            "context_precision": precision,
        },
        "quality_coverage": {
            "context_recall": quality_count,
            "context_precision": quality_count,
        },
        "runtime": {
            "latency_count": successful,
            "mean_latency_ms": p95 * 0.8,
            "p50_latency_ms": p95 * 0.7,
            "p95_latency_ms": p95,
            "max_latency_ms": p95 * 1.1,
            "application_model_calls_count": call_count,
            "average_application_model_calls": calls,
            "application_model_calls_complete": call_count == successful,
        },
        "execution": {
            "effective_config_mismatch_count": mismatches,
            "degraded_case_count": degraded,
        },
        "cases": [],
    }


@pytest.fixture
def experiment() -> dict:
    return json.loads(EXPERIMENT_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def dataset() -> dict:
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def _comparison(
    experiment: dict,
    dataset: dict,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    return build_comparison(
        experiment=experiment,
        dataset=dataset,
        manifest={"run_id": "run-test"},
        baseline=baseline,
        candidate=candidate,
    )


def test_linear_percentile_uses_interpolation() -> None:
    assert linear_percentile([10.0], 0.95) == 10.0
    assert linear_percentile([10.0, 20.0, 30.0, 40.0], 0.95) == pytest.approx(38.5)
    assert linear_percentile([], 0.95) is None


def test_runtime_metrics_exclude_warmup_and_failed_requests() -> None:
    rows = [
        _row("warmup", latency=999, calls=9, warmup=True),
        _row("ok1", latency=100, calls=1),
        _row("failed", latency=500, calls=None, error={"message": "failed"}),
        _row("ok2", latency=200, calls=1),
    ]
    scores = {
        "ok1": {"context_recall": 0.8, "context_precision": 0.9},
        "ok2": {"context_recall": 0.6, "context_precision": 0.7},
    }

    metrics = build_group_metrics(rows, scores)

    assert metrics["case_count"] == 3
    assert metrics["successful_cases"] == 2
    assert metrics["runtime"]["p95_latency_ms"] == pytest.approx(195.0)
    assert metrics["runtime"]["average_application_model_calls"] == 1.0
    assert metrics["quality"]["context_recall"] == pytest.approx(0.7)


def test_evidence_diagnostics_aggregate_traceability_and_token_usage() -> None:
    row = _row("one", latency=125, calls=2)
    row.update(
        {
            "citation_traceability": {"total": 2, "verified": 2, "rate": 1.0},
            "evidence_count": 2,
            "evidence_chars": 123,
            "evidence_status": "partial",
            "missing_aspects": ["cash"],
            "topk_outside_evidence_count": 1,
            "evidence_model_call": {
                "provider": "fake",
                "request_model": "request-model",
                "response_model": "response-model",
                "input_tokens": 20,
                "output_tokens": 5,
                "total_tokens": 25,
            },
        }
    )

    metrics = build_group_metrics(
        [row],
        {"one": {"context_recall": 0.8, "context_precision": 0.9}},
    )

    evidence = metrics["evidence"]
    assert evidence["average_evidence_count"] == 2
    assert evidence["average_evidence_chars"] == 123
    assert evidence["status_distribution"] == {"partial": 1}
    assert evidence["missing_aspects"] == {"cash": 1}
    assert evidence["topk_outside_evidence_total"] == 1
    assert evidence["citation_traceability"]["rate"] == 1.0
    assert evidence["token_usage"]["total_tokens_total"] == 25


def test_evidence_comparison_requires_nonempty_full_traceability(
    experiment: dict, dataset: dict
) -> None:
    experiment = json.loads(EVIDENCE_EXPERIMENT.read_text(encoding="utf-8"))
    baseline = _group_metrics(0.8, 0.8, 100, 1)
    candidate = _group_metrics(0.8, 0.9, 120, 2)
    candidate["evidence"] = {
        "citation_traceability": {
            "case_count": 20,
            "total_items": 0,
            "verified_items": 0,
            "rate": None,
        }
    }

    comparison = _comparison(experiment, dataset, baseline, candidate)

    assert comparison["checks"]["citation_traceability_passed"] is False
    assert comparison["verdict"] == "evaluation_failed"
    assert any("traceability" in issue for issue in comparison["issues"])

    candidate["evidence"]["citation_traceability"] = {
        "case_count": 20,
        "total_items": 35,
        "verified_items": 35,
        "rate": 1.0,
    }
    comparison = _comparison(experiment, dataset, baseline, candidate)
    assert comparison["checks"]["citation_traceability_passed"] is True


def test_complete_quality_gain_within_cost_limits_is_effective(
    experiment: dict, dataset: dict
) -> None:
    comparison = _comparison(
        experiment,
        dataset,
        _group_metrics(0.70, 0.80, 100, 0),
        _group_metrics(0.74, 0.79, 120, 1),
    )

    assert comparison["verdict"] == "effective"
    assert comparison["delta"]["context_recall"] == pytest.approx(0.04)
    assert comparison["delta"]["p95_increase_ratio"] == pytest.approx(0.2)


def test_quality_gain_over_cost_limit_is_effective_high_cost(
    experiment: dict, dataset: dict
) -> None:
    comparison = _comparison(
        experiment,
        dataset,
        _group_metrics(0.70, 0.80, 100, 0),
        _group_metrics(0.74, 0.79, 160, 1),
    )

    assert comparison["verdict"] == "effective_high_cost"


def test_primary_metric_below_threshold_is_no_clear_benefit(
    experiment: dict, dataset: dict
) -> None:
    comparison = _comparison(
        experiment,
        dataset,
        _group_metrics(0.70, 0.80, 100, 0),
        _group_metrics(0.71, 0.80, 100, 0),
    )

    assert comparison["verdict"] == "no_clear_benefit"


def test_quality_guardrail_failure_is_negative(experiment: dict, dataset: dict) -> None:
    comparison = _comparison(
        experiment,
        dataset,
        _group_metrics(0.70, 0.80, 100, 0),
        _group_metrics(0.74, 0.77, 100, 1),
    )

    assert comparison["verdict"] == "negative"


def test_incomplete_cost_or_execution_data_is_evaluation_failed(
    experiment: dict, dataset: dict
) -> None:
    candidate = _group_metrics(0.74, 0.79, 120, 1, call_count=19, mismatches=1)

    comparison = _comparison(experiment, dataset, _group_metrics(0.70, 0.80, 100, 0), candidate)

    assert comparison["verdict"] == "evaluation_failed"
    assert any("model call coverage" in issue for issue in comparison["issues"])
    assert any("configuration mismatches" in issue for issue in comparison["issues"])


def test_zero_improvement_threshold_with_flat_quality_is_not_effective(
    experiment: dict, dataset: dict
) -> None:
    experiment = copy.deepcopy(experiment)
    experiment["decision"]["min_primary_improvement"] = 0.0

    comparison = _comparison(
        experiment,
        dataset,
        _group_metrics(0.70, 0.80, 100, 0),
        _group_metrics(0.70, 0.80, 90, 0),
    )

    assert comparison["verdict"] == "no_clear_benefit"


def test_score_run_reads_saved_raw_results_without_a_retrieval_client(
    tmp_path: Path, experiment: dict, dataset: dict
) -> None:
    manifest = {"run_id": "run-test", "status": "raw_complete"}
    write_json(tmp_path / "manifest.json", manifest)
    write_json(tmp_path / "experiment.snapshot.json", experiment)
    write_json(tmp_path / "dataset.snapshot.json", dataset)
    write_jsonl(tmp_path / "baseline.raw.jsonl", [_row("one", latency=100, calls=0)])
    write_jsonl(tmp_path / "candidate.raw.jsonl", [_row("one", latency=120, calls=1)])
    scorer = StaticScorer(0.8, 0.9)

    comparison = score_run(tmp_path, scorer=scorer)

    assert scorer.calls == 2
    assert comparison["verdict"] == "no_clear_benefit"
    assert (tmp_path / "baseline.metrics.json").exists()
    assert (tmp_path / "candidate.metrics.json").exists()
    assert (tmp_path / "comparison.json").exists()
