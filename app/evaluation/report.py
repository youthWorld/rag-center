from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.evaluation.metrics import slice_quality
from app.evaluation.storage import read_jsonl, write_json


def generate_report(run_dir: Path) -> str:
    manifest = _read_json(run_dir / "manifest.json")
    experiment = _read_json(run_dir / "experiment.snapshot.json")
    dataset = _read_json(run_dir / "dataset.snapshot.json")
    comparison = _read_json(run_dir / "comparison.json")
    baseline_metrics = _read_json(run_dir / "baseline.metrics.json")
    candidate_metrics = _read_json(run_dir / "candidate.metrics.json")
    baseline_rows = read_jsonl(run_dir / "baseline.raw.jsonl")
    candidate_rows = read_jsonl(run_dir / "candidate.raw.jsonl")

    lines = [
        f"# {experiment['title']}报告",
        "",
        "## 结论",
        "",
        f"**{comparison['verdict_text']}。**",
        "",
        _conclusion_paragraph(comparison, dataset),
        "",
        *_metric_bullets(comparison),
        "",
        f"建议：{comparison['recommendation']}",
        "",
        "## 实验目标",
        "",
        f"验证 `{experiment['target']}` 在固定检索底座上的效果与成本变化。",
        "",
        "## baseline 与 candidate 的已声明差异",
        "",
        f"- baseline：{experiment['baseline']['label']}",
        f"- candidate：{experiment['candidate']['label']}",
        f"- changed_fields：{', '.join(f'`{field}`' for field in experiment['changed_fields'])}",
        "",
        "## 数据集、知识库、索引和运行条件",
        "",
        f"- run_id：`{manifest['run_id']}`",
        f"- Git commit：`{manifest['git']['commit']}`，dirty={manifest['git']['dirty']}",
        f"- 数据集：`{dataset['version']}`，suite=`{experiment['suite']}`，"
        f"正式题目数={manifest['case_count']}",
        f"- 知识库：{_format_mapping(manifest['knowledge_bases'])}",
        f"- 语料版本：{_format_mapping(manifest['corpus_versions'])}",
        f"- 并发：业务={manifest['run_config']['concurrency']}，"
        f"Judge={manifest['run_config']['ragas_concurrency']}，"
        f"预热={manifest['run_config']['warmup_cases']}",
        f"- Judge：{_judge_text(manifest.get('judge'))}",
        "- 离线观测：业务请求关闭 Langfuse，RAGAS 遥测关闭。",
        "",
        "## 公共指标对比",
        "",
        "| 指标 | baseline | candidate | 变化 | 门槛 | 是否通过 |",
        "|---|---:|---:|---:|---:|:---:|",
        *_metric_table_rows(comparison),
        "",
        "## 分题型结果",
        "",
        *_slice_lines(baseline_metrics, candidate_metrics, field="case_type"),
        "",
        "### 标签切片（至少 3 题）",
        "",
        *_slice_lines(baseline_metrics, candidate_metrics, field="tags"),
        "",
        "## 改善最多和退化最多的题目",
        "",
        *_case_change_lines(
            baseline_metrics,
            candidate_metrics,
            primary_metric=experiment["primary_metric"],
        ),
        "",
        "## 失败、降级与不可用字段",
        "",
        *_issue_lines(
            comparison,
            baseline_metrics,
            candidate_metrics,
            baseline_rows,
            candidate_rows,
        ),
        "",
        "## 成本分析",
        "",
        f"- P95 延迟：{_format_ms(comparison['baseline']['p95_latency_ms'])} → "
        f"{_format_ms(comparison['candidate']['p95_latency_ms'])}，"
        f"变化 {_format_ms_delta(comparison['delta']['p95_latency_ms'])}，"
        f"相对变化 {_format_percent(comparison['delta']['p95_increase_ratio'])}。",
        "- 平均应用模型调用数："
        f"{_format_number(comparison['baseline']['average_application_model_calls'])} → "
        f"{_format_number(comparison['candidate']['average_application_model_calls'])}，"
        f"变化 {_format_signed(comparison['delta']['average_application_model_calls'])} 次。",
        "- 应用模型调用数不包含查询 Embedding、索引构建、预热和 RAGAS Judge。",
        "",
        "## 启用或回退建议",
        "",
        comparison["recommendation"],
        "",
        "## 原始结果文件位置",
        "",
        "- `baseline.raw.jsonl`",
        "- `candidate.raw.jsonl`",
        "- `baseline.metrics.json`",
        "- `candidate.metrics.json`",
        "- `comparison.json`",
        "- `manifest.json`",
        "",
    ]
    return "\n".join(lines)


def write_report(run_dir: Path) -> Path:
    report_path = run_dir / "report.md"
    report_path.write_text(generate_report(run_dir), encoding="utf-8")
    manifest_path = run_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    manifest["status"] = "reported"
    write_json(manifest_path, manifest)
    return report_path


def _conclusion_paragraph(comparison: dict[str, Any], dataset: dict[str, Any]) -> str:
    primary = comparison["primary_metric"]
    delta = comparison["delta"].get(primary)
    if comparison["verdict"] == "evaluation_failed":
        detail = "；".join(comparison["issues"][:3]) or "关键输入不完整"
        return f"本次运行存在完整性问题（{detail}），已有数值仅供排查，不能形成启用结论。"
    return (
        f"在 `{dataset['version']}` 的 {comparison['baseline']['case_count']} 道题上，"
        f"主指标 {primary} 变化 {_format_points(delta)}。"
    )


def _metric_bullets(comparison: dict[str, Any]) -> list[str]:
    baseline = comparison["baseline"]
    candidate = comparison["candidate"]
    delta = comparison["delta"]
    return [
        "- Context Recall 从 "
        f"{_format_score(baseline['context_recall'])} 变为 "
        f"{_format_score(candidate['context_recall'])}，变化 "
        f"{_format_points(delta['context_recall'])}",
        "- Context Precision 从 "
        f"{_format_score(baseline['context_precision'])} 变为 "
        f"{_format_score(candidate['context_precision'])}，"
        f"变化 {_format_points(delta['context_precision'])}",
        f"- P95 延迟从 {_format_ms(baseline['p95_latency_ms'])} 变为 "
        f"{_format_ms(candidate['p95_latency_ms'])}，变化 "
        f"{_format_ms_delta(delta['p95_latency_ms'])}（{_format_percent(delta['p95_increase_ratio'])}）",
        "- 平均应用模型调用数从 "
        f"{_format_number(baseline['average_application_model_calls'])} 次变为 "
        f"{_format_number(candidate['average_application_model_calls'])} 次，"
        f"变化 {_format_signed(delta['average_application_model_calls'])} 次",
    ]


def _metric_table_rows(comparison: dict[str, Any]) -> list[str]:
    baseline = comparison["baseline"]
    candidate = comparison["candidate"]
    delta = comparison["delta"]
    thresholds = comparison["thresholds"]
    checks = comparison["checks"]
    precision_limit = thresholds["quality_guardrails"]["context_precision"][
        "min_delta"
    ]
    return [
        "| Context Recall | "
        f"{_format_score(baseline['context_recall'])} | "
        f"{_format_score(candidate['context_recall'])} | "
        f"{_format_points(delta['context_recall'])} | "
        f"≥ {_format_points(thresholds['min_primary_improvement'])} | "
        f"{_yes_no(checks['primary_threshold_passed'])} |",
        "| Context Precision | "
        f"{_format_score(baseline['context_precision'])} | "
        f"{_format_score(candidate['context_precision'])} | "
        f"{_format_points(delta['context_precision'])} | "
        f"≥ {_format_points(precision_limit)} | "
        f"{_yes_no(checks['quality_guardrails'].get('context_precision', False))} |",
        "| P95 延迟 | "
        f"{_format_ms(baseline['p95_latency_ms'])} | "
        f"{_format_ms(candidate['p95_latency_ms'])} | "
        f"{_format_ms_delta(delta['p95_latency_ms'])} / "
        f"{_format_percent(delta['p95_increase_ratio'])} | "
        f"≤ {_format_percent(thresholds['cost_limits']['max_p95_increase_ratio'])} | "
        f"{_yes_no(checks['p95_cost_passed'])} |",
        "| 平均应用模型调用数 | "
        f"{_format_number(baseline['average_application_model_calls'])} | "
        f"{_format_number(candidate['average_application_model_calls'])} | "
        f"{_format_signed(delta['average_application_model_calls'])} 次 | "
        f"≤ +{thresholds['cost_limits']['max_model_calls_increase']:.1f} 次 | "
        f"{_yes_no(checks['model_call_cost_passed'])} |",
    ]


def _slice_lines(
    baseline: dict[str, Any], candidate: dict[str, Any], *, field: str
) -> list[str]:
    left = slice_quality(baseline, field=field)
    right = slice_quality(candidate, field=field)
    values = sorted(set(left) | set(right))
    if not values:
        return ["没有达到至少 3 题展示门槛的切片。"]
    lines = [
        "| 切片 | 题数 | baseline Recall | candidate Recall | Recall 变化 |",
        "|---|---:|---:|---:|---:|",
    ]
    for value in values:
        baseline_slice = left.get(value, {})
        candidate_slice = right.get(value, {})
        recall_delta = _difference(
            candidate_slice.get("context_recall"),
            baseline_slice.get("context_recall"),
        )
        lines.append(
            f"| {value} | {max(baseline_slice.get('count', 0), candidate_slice.get('count', 0))} "
            f"| {_format_score(baseline_slice.get('context_recall'))} "
            f"| {_format_score(candidate_slice.get('context_recall'))} "
            f"| {_format_points(recall_delta)} |"
        )
    return lines


def _case_change_lines(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    primary_metric: str,
) -> list[str]:
    baseline_cases = {case["case_id"]: case for case in baseline["cases"]}
    changes: list[tuple[float, str]] = []
    for case in candidate["cases"]:
        left = baseline_cases.get(case["case_id"], {})
        delta = _difference(case.get(primary_metric), left.get(primary_metric))
        if delta is not None:
            changes.append((delta, case["case_id"]))
    if not changes:
        return ["没有可比较的逐题质量分数。"]
    improvements = sorted(changes, reverse=True)[:3]
    regressions = sorted(changes)[:3]
    lines = ["### 改善最多", ""]
    lines.extend(f"- `{case_id}`：{_format_points(delta)}" for delta, case_id in improvements)
    lines.extend(["", "### 退化最多", ""] )
    lines.extend(f"- `{case_id}`：{_format_points(delta)}" for delta, case_id in regressions)
    return lines


def _issue_lines(
    comparison: dict[str, Any],
    baseline_metrics: dict[str, Any],
    candidate_metrics: dict[str, Any],
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> list[str]:
    lines: list[str] = []
    if comparison["issues"]:
        lines.extend(f"- {issue}" for issue in comparison["issues"])
    for group_name, metrics, rows in (
        ("baseline", baseline_metrics, baseline_rows),
        ("candidate", candidate_metrics, candidate_rows),
    ):
        failed_ids = [case["case_id"] for case in metrics["cases"] if case["retrieve_error"]]
        degraded_ids = [row["case_id"] for row in rows if row.get("degradation")]
        mismatch_ids = [
            row["case_id"] for row in rows if row.get("effective_config_mismatches")
        ]
        if failed_ids:
            lines.append(f"- {group_name} 请求失败：{', '.join(failed_ids)}")
        if degraded_ids:
            lines.append(f"- {group_name} 降级：{', '.join(degraded_ids)}")
        if mismatch_ids:
            lines.append(f"- {group_name} 实际配置不一致：{', '.join(mismatch_ids)}")
    if not lines:
        return ["- 无请求失败、评分缺失、目标能力降级或实际配置不一致。"]
    return lines


def _judge_text(value: Any) -> str:
    if not isinstance(value, dict):
        return "未记录"
    return (
        f"model={value.get('model') or 'unknown'}，temperature={value.get('temperature')}，"
        f"timeout={value.get('timeout_seconds')}s，max_workers={value.get('max_workers')}"
    )


def _format_mapping(value: dict[str, Any]) -> str:
    return "，".join(f"{key}=`{item}`" for key, item in sorted(value.items()))


def _format_score(value: Any) -> str:
    return f"{float(value):.3f}" if _number(value) else "不可用"


def _format_points(value: Any) -> str:
    return f"{float(value) * 100:+.1f} 个百分点" if _number(value) else "不可用"


def _format_ms(value: Any) -> str:
    return f"{float(value):.1f} ms" if _number(value) else "不可用"


def _format_ms_delta(value: Any) -> str:
    return f"{float(value):+.1f} ms" if _number(value) else "不可用"


def _format_percent(value: Any) -> str:
    return f"{float(value) * 100:+.1f}%" if _number(value) else "不可用"


def _format_number(value: Any) -> str:
    return f"{float(value):.2f}" if _number(value) else "不可用"


def _format_signed(value: Any) -> str:
    return f"{float(value):+.2f}" if _number(value) else "不可用"


def _yes_no(value: bool) -> str:
    return "是" if value else "否"


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _difference(candidate: Any, baseline: Any) -> float | None:
    if not _number(candidate) or not _number(baseline):
        return None
    return float(candidate) - float(baseline)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"required evaluation file does not exist: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"evaluation file must contain an object: {path}")
    return payload
