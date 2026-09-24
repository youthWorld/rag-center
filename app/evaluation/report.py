from __future__ import annotations

import json
import math
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
        f"题目数={manifest['case_count']}，"
        f"运行类型={_run_type(experiment, manifest)}",
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
        *(
            _rerank_diagnostics(experiment, dataset, baseline_rows, candidate_rows)
            if experiment.get("rerank_experiment") == "effect"
            else []
        ),
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
        *(["- `shared_candidates.raw.jsonl`"] if experiment.get("rerank_experiment") else []),
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


def _rerank_diagnostics(
    experiment: dict[str, Any],
    dataset: dict[str, Any],
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> list[str]:
    del experiment
    cases = {case["id"]: case for case in dataset["cases"]}

    def score(rows: list[dict[str, Any]]) -> float | None:
        values: list[float] = []
        for row in rows:
            if row.get("error") or row["case_id"] not in cases:
                return None
            case = cases[row["case_id"]]
            docs = [doc.removesuffix(".md").lower() for doc in case.get("expected_documents", [])]
            headings = [str(h).lower() for h in case.get("expected_headings", [])]
            if not docs and not headings:
                return None
            gains = []
            for context in row["contexts"][:10]:
                document = str(context.get("document") or "").removesuffix(".md").lower()
                heading = str(context.get("heading") or "").lower()
                relevant = (not docs or any(doc in document for doc in docs)) and (
                    not headings or any(h in heading for h in headings)
                )
                gains.append(int(relevant))
            dcg = sum(gain / math.log2(rank + 2) for rank, gain in enumerate(gains))
            ideal_count = min(10, max(len(docs), len(headings), 1))
            ideal = sum(1 / math.log2(rank + 2) for rank in range(ideal_count))
            values.append(dcg / ideal)
        return sum(values) / len(values) if values else None

    left, right = score(baseline_rows), score(candidate_rows)
    return [
        "## 排序诊断（非正式验收指标）",
        "",
        "nDCG@10 使用 Golden Set 期望文档及章节的二元相关性，"
        "仅用于定位排序变化，不参与四项正式指标的判定。",
        f"- baseline nDCG@10：{_format_score(left)}",
        f"- candidate nDCG@10：{_format_score(right)}",
        "",
    ]


def _run_type(experiment: dict[str, Any], manifest: dict[str, Any]) -> str:
    if experiment.get("rerank_experiment") and manifest["case_count"] != 20:
        return "预检（非正式结论）"
    return "正式评测"


def _conclusion_paragraph(comparison: dict[str, Any], dataset: dict[str, Any]) -> str:
    primary = comparison["primary_metric"]
    delta = comparison["delta"].get(primary)
    if comparison["verdict"] == "preflight_only":
        return "少量样本仅验证候选冻结、精排、RAGAS 和报告生成链路；指标不作为正式效果结论。"
    if comparison["verdict"] == "evaluation_failed":
        detail = "；".join(comparison["issues"][:3]) or "关键输入不完整"
        return f"本次运行存在完整性问题（{detail}），已有数值仅供排查，不能形成启用结论。"
    conclusion = (
        f"在 `{dataset['version']}` 的 {comparison['baseline']['case_count']} 道题上，"
        f"主指标 {primary} 变化 {_format_points(delta)}。"
    )
    if comparison["experiment_id"].startswith("rerank_"):
        recall_delta = comparison["delta"].get("context_recall")
        precision_delta = comparison["delta"].get("context_precision")
        latency_delta = comparison["delta"].get("p95_latency_ms")
        calls_delta = comparison["delta"].get("average_application_model_calls")
        if recall_delta is not None and precision_delta is not None:
            if recall_delta * precision_delta < 0:
                conclusion += "两项质量指标方向相反，不能宣称质量全面提升；"
            elif recall_delta >= 0 and precision_delta >= 0:
                conclusion += "两项质量指标未下降；"
            else:
                conclusion += "质量指标存在退化；"
        conclusion += (
            f"Context Recall 变化 {_format_points(recall_delta)}，"
            f"Context Precision 变化 {_format_points(precision_delta)}，"
            f"P95 变化 {_format_ms_delta(latency_delta)}，"
            f"平均应用模型调用变化 {_format_signed(calls_delta)} 次。"
            "本轮仅有 20 题，需结合逐题退化及成本上限决定是否扩大验证，"
            "不能把历史模型数据当作本模型结论。"
        )
    return conclusion


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
    guard_metric = next(iter(thresholds["quality_guardrails"]))
    guard_limit = thresholds["quality_guardrails"][guard_metric]["min_delta"]
    primary = comparison["primary_metric"]
    labels = {"context_recall": "Context Recall", "context_precision": "Context Precision"}
    quality_rows = []
    for metric in ("context_recall", "context_precision"):
        limit = thresholds["min_primary_improvement"] if metric == primary else guard_limit
        passed = (
            checks["primary_threshold_passed"]
            if metric == primary
            else checks["quality_guardrails"].get(metric, False)
        )
        quality_rows.append(
            f"| {labels[metric]} | {_format_score(baseline[metric])} | "
            f"{_format_score(candidate[metric])} | {_format_points(delta[metric])} | "
            f"≥ {_format_points(limit)} | {_yes_no(passed)} |"
        )
    return [
        *quality_rows,
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


def _slice_lines(baseline: dict[str, Any], candidate: dict[str, Any], *, field: str) -> list[str]:
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
    lines.extend(["", "### 退化最多", ""])
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
        mismatch_ids = [row["case_id"] for row in rows if row.get("effective_config_mismatches")]
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
