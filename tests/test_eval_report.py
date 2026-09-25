from __future__ import annotations

import json
from pathlib import Path

from app.evaluation.metrics import build_comparison
from app.evaluation.report import generate_report, write_report
from app.evaluation.storage import write_json, write_jsonl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_PATH = PROJECT_ROOT / "eval" / "experiments" / "query_rewrite.json"
EVIDENCE_EXPERIMENT_PATH = PROJECT_ROOT / "eval" / "experiments" / "evidence_orchestration.json"
DATASET_PATH = PROJECT_ROOT / "eval" / "datasets" / "golden_basic_20.json"


def _metrics(recall: float, precision: float, p95: float, calls: float) -> dict:
    cases = [
        {
            "case_id": f"case_{index}",
            "case_type": "direct_fact",
            "tags": ["tag"],
            "context_recall": recall + index / 1000,
            "context_precision": precision,
            "score_error": None,
            "retrieve_error": None,
        }
        for index in range(3)
    ]
    return {
        "case_count": 3,
        "successful_cases": 3,
        "failed_cases": 0,
        "quality": {"context_recall": recall, "context_precision": precision},
        "quality_coverage": {"context_recall": 3, "context_precision": 3},
        "runtime": {
            "latency_count": 3,
            "mean_latency_ms": p95 - 10,
            "p50_latency_ms": p95 - 5,
            "p95_latency_ms": p95,
            "max_latency_ms": p95 + 5,
            "application_model_calls_count": 3,
            "average_application_model_calls": calls,
            "application_model_calls_complete": True,
        },
        "execution": {
            "effective_config_mismatch_count": 0,
            "degraded_case_count": 0,
        },
        "cases": cases,
    }


def _raw_rows() -> list[dict]:
    return [
        {
            "case_id": f"case_{index}",
            "effective_config_mismatches": [],
            "degradation": [],
            "error": None,
        }
        for index in range(3)
    ]


def test_report_uses_comparison_values_and_starts_with_explicit_conclusion(
    tmp_path: Path,
) -> None:
    experiment = json.loads(EXPERIMENT_PATH.read_text(encoding="utf-8"))
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    baseline = _metrics(0.700, 0.800, 100, 0)
    candidate = _metrics(0.740, 0.790, 120, 1)
    manifest = {
        "run_id": "run-test",
        "status": "scored",
        "git": {"commit": "abc123", "dirty": False},
        "case_count": 3,
        "knowledge_bases": {"ecommerce": "kb-test"},
        "corpus_versions": {"ecommerce": "corpus-test"},
        "run_config": {
            "concurrency": 4,
            "ragas_concurrency": 4,
            "warmup_cases": 0,
        },
        "judge": {
            "model": "judge-test",
            "temperature": 0.0,
            "timeout_seconds": 120,
            "max_workers": 4,
        },
    }
    comparison = build_comparison(
        experiment=experiment,
        dataset=dataset,
        manifest=manifest,
        baseline=baseline,
        candidate=candidate,
    )
    write_json(tmp_path / "manifest.json", manifest)
    write_json(tmp_path / "experiment.snapshot.json", experiment)
    write_json(tmp_path / "dataset.snapshot.json", dataset)
    write_json(tmp_path / "comparison.json", comparison)
    write_json(tmp_path / "baseline.metrics.json", baseline)
    write_json(tmp_path / "candidate.metrics.json", candidate)
    write_jsonl(tmp_path / "baseline.raw.jsonl", _raw_rows())
    write_jsonl(tmp_path / "candidate.raw.jsonl", _raw_rows())

    report = generate_report(tmp_path)

    assert report.startswith("# Query Rewrite 消融实验报告\n\n## 结论")
    assert "**有效，建议启用。**" in report
    assert "0.700" in report
    assert "0.740" in report
    assert "+4.0 个百分点" in report
    assert "+20.0 ms" in report
    assert "+20.0%" in report
    assert "+1.00 次" in report

    report_path = write_report(tmp_path)
    assert report_path.read_text(encoding="utf-8") == report
    saved_manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert saved_manifest["status"] == "reported"


def test_evidence_report_keeps_four_formal_metrics_and_adds_traceability_diagnostics(
    tmp_path: Path,
) -> None:
    experiment = json.loads(EVIDENCE_EXPERIMENT_PATH.read_text(encoding="utf-8"))
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    baseline = _metrics(0.80, 0.80, 100, 1)
    candidate = _metrics(0.80, 0.90, 125, 2)
    candidate["evidence"] = {
        "average_evidence_count": 2.0,
        "average_evidence_chars": 320.0,
        "status_distribution": {"complete": 3},
        "missing_aspects": {},
        "topk_outside_evidence_total": 1.0,
        "average_topk_outside_evidence_count": 1 / 3,
        "citation_traceability": {
            "case_count": 3,
            "total_items": 6,
            "verified_items": 6,
            "rate": 1.0,
        },
        "token_usage": {
            "request_models": ["evidence-model"],
            "response_models": ["evidence-model-v2"],
            "input_tokens_total": 120,
            "output_tokens_total": 30,
            "total_tokens_total": 150,
            "average_total_tokens": 50,
        },
    }
    manifest = {
        "run_id": "run-evidence",
        "status": "scored",
        "git": {"commit": "abc123", "dirty": False},
        "case_count": 3,
        "knowledge_bases": {"ecommerce": "kb-test"},
        "corpus_versions": {"ecommerce": "corpus-test"},
        "run_config": {
            "concurrency": 4,
            "ragas_concurrency": 4,
            "warmup_cases": 0,
        },
        "judge": {
            "model": "judge-test",
            "temperature": 0.0,
            "timeout_seconds": 120,
            "max_workers": 4,
        },
    }
    comparison = build_comparison(
        experiment=experiment,
        dataset=dataset,
        manifest=manifest,
        baseline=baseline,
        candidate=candidate,
    )
    write_json(tmp_path / "manifest.json", manifest)
    write_json(tmp_path / "experiment.snapshot.json", experiment)
    write_json(tmp_path / "dataset.snapshot.json", dataset)
    write_json(tmp_path / "comparison.json", comparison)
    write_json(tmp_path / "baseline.metrics.json", baseline)
    write_json(tmp_path / "candidate.metrics.json", candidate)
    write_jsonl(tmp_path / "baseline.raw.jsonl", _raw_rows())
    write_jsonl(tmp_path / "candidate.raw.jsonl", _raw_rows())

    report = generate_report(tmp_path)

    assert report.count("| Context Recall |") == 1
    assert report.count("| Context Precision |") == 1
    assert report.count("| P95 延迟 |") == 1
    assert report.count("| 平均应用模型调用数 |") == 1
    assert "专题硬校验与诊断（非第五项正式指标）" in report
    assert "引用回溯率：100.0%（6/6，硬门槛 100%，通过）" in report
    assert "total 总计 150.00" in report
    assert "Provider 层不固化价格" in report
    assert "- `shared_pipeline.raw.jsonl`" in report
