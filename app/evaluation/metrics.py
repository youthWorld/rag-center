from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Protocol

from app.core.config import Settings
from app.evaluation.storage import read_jsonl, write_json

QUALITY_METRICS = ("context_recall", "context_precision")
VERDICT_TEXT = {
    "preflight_only": "预检完成，不形成正式结论",
    "effective": "有效，建议启用",
    "effective_high_cost": "有效但成本较高，建议按场景启用",
    "no_clear_benefit": "无明显收益，暂不启用",
    "negative": "出现负向效果，建议回退",
    "evaluation_failed": "评测执行失败，暂时不能下结论",
}


class QualityScorer(Protocol):
    metadata: dict[str, Any]

    def score(self, rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]: ...


class RagasQualityScorer:
    def __init__(self, *, max_workers: int, timeout_seconds: int) -> None:
        self.max_workers = max_workers
        self.timeout_seconds = timeout_seconds
        self.metadata: dict[str, Any] = {
            "provider": "openai_compatible",
            "model": None,
            "temperature": 0.0,
            "timeout_seconds": timeout_seconds,
            "max_workers": max_workers,
            "max_retries": 2,
            "seed": 42,
        }

    def score(self, rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        if not rows:
            return {}
        os.environ["RAGAS_DO_NOT_TRACK"] = "true"
        try:
            from datasets import Dataset
            from openai import OpenAI
            from ragas import RunConfig, evaluate
            from ragas.llms import llm_factory

            # The project lock currently uses RAGAS 0.4.3. Its legacy
            # evaluate() entry point requires the legacy singleton Metric
            # objects; the newer collections classes are incompatible with
            # that entry point even though their names are similar.
            from ragas.metrics import context_precision, context_recall

            settings = Settings()
            api_key, base_url = _judge_credentials(settings)
            model = (
                os.getenv("RAGAS_LLM_MODEL")
                or os.getenv("EVAL_MODEL")
                or settings.llm_model
            )
            self.metadata.update(
                {
                    "model": model,
                    "base_url_origin": _safe_origin(base_url),
                }
            )
            client = OpenAI(api_key=api_key, base_url=base_url)
            evaluator_llm = llm_factory(
                model,
                client=client,
                temperature=0.0,
            )
            dataset = Dataset.from_list(
                [
                    {
                        "user_input": row["question"],
                        "reference": row["ground_truth"],
                        "retrieved_contexts": [
                            context["content"] for context in row["contexts"]
                        ],
                    }
                    for row in rows
                ]
            )
            result = evaluate(
                dataset,
                metrics=[context_recall, context_precision],
                llm=evaluator_llm,
                run_config=RunConfig(
                    max_workers=self.max_workers,
                    timeout=self.timeout_seconds,
                    max_retries=2,
                    seed=42,
                ),
                raise_exceptions=False,
                show_progress=False,
            )
            records = _result_records(result)
            if len(records) != len(rows):
                raise RuntimeError(
                    f"RAGAS returned {len(records)} rows for {len(rows)} cases"
                )
            return {
                row["case_id"]: _normalize_quality_record(record)
                for row, record in zip(rows, records, strict=True)
            }
        except Exception as exc:
            message = str(exc) or type(exc).__name__
            return {
                row["case_id"]: {
                    "context_recall": None,
                    "context_precision": None,
                    "error": {"type": type(exc).__name__, "message": message},
                }
                for row in rows
            }


def linear_percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    fraction = position - lower_index
    return ordered[lower_index] + (ordered[upper_index] - ordered[lower_index]) * fraction


def build_group_metrics(
    rows: list[dict[str, Any]],
    quality_scores: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    formal_rows = [row for row in rows if not row.get("warmup", False)]
    successful_rows = [row for row in formal_rows if row.get("error") is None]
    latencies = [
        float(row["latency_ms"])
        for row in successful_rows
        if _finite_number(row.get("latency_ms"))
    ]
    model_calls = [
        float(row["application_model_calls"])
        for row in successful_rows
        if _finite_number(row.get("application_model_calls"))
    ]

    cases: list[dict[str, Any]] = []
    quality_values: dict[str, list[float]] = defaultdict(list)
    for row in formal_rows:
        score = quality_scores.get(row["case_id"], {})
        case_metrics = {
            metric: _finite_or_none(score.get(metric)) for metric in QUALITY_METRICS
        }
        if row.get("error") is None:
            for metric, value in case_metrics.items():
                if value is not None:
                    quality_values[metric].append(value)
        cases.append(
            {
                "case_id": row["case_id"],
                "case_type": row.get("case_type"),
                "tags": row.get("tags", []),
                **case_metrics,
                "score_error": score.get("error"),
                "retrieve_error": row.get("error"),
            }
        )

    return {
        "case_count": len(formal_rows),
        "successful_cases": len(successful_rows),
        "failed_cases": len(formal_rows) - len(successful_rows),
        "quality": {
            metric: _mean_or_none(quality_values[metric]) for metric in QUALITY_METRICS
        },
        "quality_coverage": {
            metric: len(quality_values[metric]) for metric in QUALITY_METRICS
        },
        "runtime": {
            "latency_count": len(latencies),
            "mean_latency_ms": _mean_or_none(latencies),
            "p50_latency_ms": linear_percentile(latencies, 0.50),
            "p95_latency_ms": linear_percentile(latencies, 0.95),
            "max_latency_ms": max(latencies) if latencies else None,
            "application_model_calls_count": len(model_calls),
            "average_application_model_calls": _mean_or_none(model_calls),
            "application_model_calls_complete": len(model_calls) == len(successful_rows),
        },
        "execution": {
            "effective_config_mismatch_count": sum(
                bool(row.get("effective_config_mismatches")) for row in successful_rows
            ),
            "degraded_case_count": sum(
                bool(row.get("degradation")) for row in successful_rows
            ),
        },
        "cases": cases,
    }


def build_comparison(
    *,
    experiment: dict[str, Any],
    dataset: dict[str, Any],
    manifest: dict[str, Any],
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    baseline_summary = _comparison_group_summary(baseline)
    candidate_summary = _comparison_group_summary(candidate)
    delta = {
        "context_recall": _difference(
            candidate_summary["context_recall"], baseline_summary["context_recall"]
        ),
        "context_precision": _difference(
            candidate_summary["context_precision"],
            baseline_summary["context_precision"],
        ),
        "p95_latency_ms": _difference(
            candidate_summary["p95_latency_ms"], baseline_summary["p95_latency_ms"]
        ),
        "p95_increase_ratio": _increase_ratio(
            candidate_summary["p95_latency_ms"], baseline_summary["p95_latency_ms"]
        ),
        "average_application_model_calls": _difference(
            candidate_summary["average_application_model_calls"],
            baseline_summary["average_application_model_calls"],
        ),
    }
    checks, issues = _evaluate_checks(
        experiment=experiment,
        baseline=baseline,
        candidate=candidate,
        delta=delta,
    )
    verdict = _decide_verdict(checks)
    if experiment.get("rerank_experiment") and manifest["case_count"] != 20 and checks["complete"]:
        verdict = "preflight_only"
    return {
        "schema_version": "1.0",
        "experiment_id": experiment["experiment_id"],
        "dataset_version": dataset["version"],
        "suite": experiment["suite"],
        "run_id": manifest["run_id"],
        "verdict": verdict,
        "verdict_text": VERDICT_TEXT[verdict],
        "primary_metric": experiment["primary_metric"],
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "delta": delta,
        "thresholds": experiment["decision"],
        "checks": checks,
        "issues": issues,
        "failed_cases": {
            "baseline": baseline["failed_cases"],
            "candidate": candidate["failed_cases"],
        },
        "recommendation": _recommendation(verdict),
    }


def score_run(
    run_dir: Path,
    *,
    scorer: QualityScorer | None = None,
) -> dict[str, Any]:
    manifest = _read_json(run_dir / "manifest.json")
    experiment = _read_json(run_dir / "experiment.snapshot.json")
    dataset = _read_json(run_dir / "dataset.snapshot.json")
    scorer = scorer or RagasQualityScorer(
        max_workers=experiment["ragas_concurrency"],
        timeout_seconds=experiment["timeout_seconds"],
    )

    group_metrics: dict[str, dict[str, Any]] = {}
    for group_name in ("baseline", "candidate"):
        rows = read_jsonl(run_dir / f"{group_name}.raw.jsonl")
        successful_rows = [row for row in rows if row.get("error") is None]
        scores = scorer.score(successful_rows)
        metrics = build_group_metrics(rows, scores)
        group_metrics[group_name] = metrics
        write_json(run_dir / f"{group_name}.metrics.json", metrics)

    manifest["judge"] = scorer.metadata
    comparison = build_comparison(
        experiment=experiment,
        dataset=dataset,
        manifest=manifest,
        baseline=group_metrics["baseline"],
        candidate=group_metrics["candidate"],
    )
    write_json(run_dir / "comparison.json", comparison)
    manifest["status"] = "scored"
    write_json(run_dir / "manifest.json", manifest)
    return comparison


def slice_quality(
    metrics: dict[str, Any], *, field: str, minimum_size: int = 3
) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in metrics["cases"]:
        raw_values = case.get(field)
        values = raw_values if isinstance(raw_values, list) else [raw_values]
        for value in values:
            if isinstance(value, str) and value:
                buckets[value].append(case)
    result: dict[str, dict[str, Any]] = {}
    for value, cases in sorted(buckets.items()):
        if len(cases) < minimum_size:
            continue
        result[value] = {
            "count": len(cases),
            **{
                metric: _mean_or_none(
                    [case[metric] for case in cases if _finite_number(case.get(metric))]
                )
                for metric in QUALITY_METRICS
            },
        }
    return result


def _comparison_group_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_count": metrics["case_count"],
        "successful_cases": metrics["successful_cases"],
        "failed_cases": metrics["failed_cases"],
        "context_recall": metrics["quality"]["context_recall"],
        "context_precision": metrics["quality"]["context_precision"],
        "mean_latency_ms": metrics["runtime"]["mean_latency_ms"],
        "p50_latency_ms": metrics["runtime"]["p50_latency_ms"],
        "p95_latency_ms": metrics["runtime"]["p95_latency_ms"],
        "max_latency_ms": metrics["runtime"]["max_latency_ms"],
        "average_application_model_calls": metrics["runtime"][
            "average_application_model_calls"
        ],
        "application_model_calls_count": metrics["runtime"][
            "application_model_calls_count"
        ],
    }


def _evaluate_checks(
    *,
    experiment: dict[str, Any],
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    delta: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    decision = experiment["decision"]
    max_failed = decision["max_failed_cases"]
    issues: list[str] = []
    expected_cases = baseline["case_count"]
    for name, metrics in (("baseline", baseline), ("candidate", candidate)):
        if metrics["case_count"] != expected_cases:
            issues.append(f"{name} case count differs from baseline")
        if metrics["failed_cases"] > max_failed:
            issues.append(
                f"{name} failed cases {metrics['failed_cases']} exceed limit {max_failed}"
            )
        for metric in QUALITY_METRICS:
            if metrics["quality_coverage"][metric] != metrics["successful_cases"]:
                issues.append(f"{name} {metric} coverage is incomplete")
        if metrics["runtime"]["latency_count"] != metrics["successful_cases"]:
            issues.append(f"{name} latency coverage is incomplete")
        if not metrics["runtime"]["application_model_calls_complete"]:
            issues.append(f"{name} application model call coverage is incomplete")
        if metrics["execution"]["effective_config_mismatch_count"]:
            issues.append(f"{name} contains effective configuration mismatches")
        if metrics["execution"]["degraded_case_count"]:
            issues.append(f"{name} contains degraded target executions")

    complete = not issues
    primary_metric = experiment["primary_metric"]
    primary_delta = delta[primary_metric]
    min_improvement = decision["min_primary_improvement"]
    guardrails = {
        metric: (
            delta.get(metric) is not None
            and delta[metric] >= config["min_delta"]
        )
        for metric, config in decision["quality_guardrails"].items()
    }
    primary_negative = (
        primary_delta is not None
        and (
            primary_delta <= -min_improvement
            if min_improvement > 0
            else primary_delta < 0
        )
    )
    primary_threshold_passed = (
        primary_delta is not None and primary_delta >= min_improvement
    )
    quality_positive = any(
        delta.get(metric) is not None and delta[metric] > 0
        for metric in {primary_metric, *decision["quality_guardrails"].keys()}
    )
    p95_ratio = delta["p95_increase_ratio"]
    model_call_delta = delta["average_application_model_calls"]
    cost_limits = decision["cost_limits"]
    p95_cost_passed = (
        p95_ratio is not None
        and p95_ratio <= cost_limits["max_p95_increase_ratio"]
    )
    model_call_cost_passed = (
        model_call_delta is not None
        and model_call_delta <= cost_limits["max_model_calls_increase"]
    )
    return (
        {
            "complete": complete,
            "quality_guardrails": guardrails,
            "quality_guardrails_passed": all(guardrails.values()),
            "primary_negative": primary_negative,
            "primary_threshold_passed": primary_threshold_passed,
            "quality_positive": quality_positive,
            "p95_cost_passed": p95_cost_passed,
            "model_call_cost_passed": model_call_cost_passed,
            "cost_limits_passed": p95_cost_passed and model_call_cost_passed,
        },
        issues,
    )


def _decide_verdict(checks: dict[str, Any]) -> str:
    if not checks["complete"]:
        return "evaluation_failed"
    if checks["primary_negative"] or not checks["quality_guardrails_passed"]:
        return "negative"
    if not checks["primary_threshold_passed"] or not checks["quality_positive"]:
        return "no_clear_benefit"
    if not checks["cost_limits_passed"]:
        return "effective_high_cost"
    return "effective"


def _recommendation(verdict: str) -> str:
    return {
        "effective": "在本实验固定条件下建议启用；不自动修改线上预设。",
        "preflight_only": "仅验证执行链路，须完成 20 题正式实验后再判断效果。",
        "effective_high_cost": "仅建议在质量优先或复杂问题场景启用。",
        "no_clear_benefit": "当前数据集未证明稳定收益，暂不调整默认配置。",
        "negative": "候选方案触发质量回归，建议保持或恢复基线配置。",
        "evaluation_failed": "先修复运行完整性问题并新建运行目录重跑。",
    }[verdict]


def _judge_credentials(settings: Settings) -> tuple[str, str]:
    if settings.model_api_key:
        return settings.model_api_key, settings.model_base_url
    if settings.llm_api_key:
        return settings.llm_api_key, settings.llm_base_url
    raise RuntimeError("MODEL_API_KEY or LLM_API_KEY is required for RAGAS scoring")


def _result_records(result: Any) -> list[dict[str, Any]]:
    scores = getattr(result, "scores", None)
    if isinstance(scores, list) and all(isinstance(item, dict) for item in scores):
        return list(scores)
    to_pandas = getattr(result, "to_pandas", None)
    if callable(to_pandas):
        records = to_pandas().to_dict(orient="records")
        if isinstance(records, list) and all(isinstance(item, dict) for item in records):
            return records
    if isinstance(result, dict):
        return [result]
    raise RuntimeError("RAGAS returned no per-case score records")


def _normalize_quality_record(record: dict[str, Any]) -> dict[str, Any]:
    values = {metric: _finite_or_none(record.get(metric)) for metric in QUALITY_METRICS}
    missing = [metric for metric, value in values.items() if value is None]
    values["error"] = (
        {"type": "ScoreUnavailable", "message": f"missing scores: {missing}"}
        if missing
        else None
    )
    return values


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"required evaluation file does not exist: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"evaluation file must contain an object: {path}")
    return payload


def _safe_origin(base_url: str) -> str:
    try:
        from urllib.parse import urlsplit

        parsed = urlsplit(base_url)
        return f"{parsed.scheme}://{parsed.netloc}" if parsed.netloc else "configured"
    except Exception:
        return "configured"


def _mean_or_none(values: list[float]) -> float | None:
    return fmean(values) if values else None


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _finite_or_none(value: Any) -> float | None:
    return float(value) if _finite_number(value) else None


def _difference(candidate: Any, baseline: Any) -> float | None:
    if not _finite_number(candidate) or not _finite_number(baseline):
        return None
    return float(candidate) - float(baseline)


def _increase_ratio(candidate: Any, baseline: Any) -> float | None:
    if not _finite_number(candidate) or not _finite_number(baseline):
        return None
    if float(baseline) == 0:
        return 0.0 if float(candidate) == 0 else None
    return (float(candidate) - float(baseline)) / float(baseline)
