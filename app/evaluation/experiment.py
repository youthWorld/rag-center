from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.tenant.retrieve_presets import expand_retrieve_profile

GROUP_NAMES = ("baseline", "candidate")
NAMED_PROFILES = {"speed", "balanced", "quality"}
SUPPORTED_PROFILES = NAMED_PROFILES | {"custom"}
SUPPORTED_PRIMARY_METRICS = {"context_recall", "context_precision"}
GROUP_REQUEST_FIELDS = {
    "profile",
    "top_k",
    "retrieval_options",
    "rerank_options",
    "query_options",
}


class ExperimentValidationError(ValueError):
    """Raised when an experiment cannot be safely compared or executed."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("experiment validation failed:\n" + "\n".join(f"- {e}" for e in errors))


def load_experiment(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ExperimentValidationError([f"experiment does not exist: {path}"]) from exc
    except OSError as exc:
        raise ExperimentValidationError([f"failed to read experiment {path}: {exc}"]) from exc
    except json.JSONDecodeError as exc:
        raise ExperimentValidationError(
            [
                "experiment contains invalid JSON at "
                f"line {exc.lineno}, column {exc.colno}: {exc.msg}"
            ]
        ) from exc
    if not isinstance(payload, dict):
        raise ExperimentValidationError(["root must be a JSON object"])
    return payload


def load_and_validate_experiment(path: Path) -> dict[str, Any]:
    payload = load_experiment(path)
    validate_experiment(payload)
    return payload


def validate_experiment(payload: dict[str, Any]) -> None:
    errors: list[str] = []
    for field in ("schema_version", "experiment_id", "title", "target", "dataset"):
        _require_text(payload, field, errors, location="root")

    suite = _require_text(payload, "suite", errors, location="root")
    if suite and suite != "basic":
        errors.append("root.suite must be 'basic' for optimization fifteen experiments")

    primary_metric = _require_text(payload, "primary_metric", errors, location="root")
    if primary_metric and primary_metric not in SUPPORTED_PRIMARY_METRICS:
        errors.append(
            f"root.primary_metric must be one of {sorted(SUPPORTED_PRIMARY_METRICS)}"
        )

    for field in ("concurrency", "ragas_concurrency", "timeout_seconds"):
        _require_positive_int(payload, field, errors, location="root")
    warmup_cases = payload.get("warmup_cases")
    if not isinstance(warmup_cases, int) or isinstance(warmup_cases, bool) or warmup_cases < 0:
        errors.append("root.warmup_cases must be a non-negative integer")

    changed_fields = payload.get("changed_fields")
    if not isinstance(changed_fields, list) or not changed_fields:
        errors.append("root.changed_fields must be a non-empty array")
        changed_fields = []
    else:
        for index, field in enumerate(changed_fields):
            if not isinstance(field, str) or not field.strip():
                errors.append(f"root.changed_fields[{index}] must be a non-empty string")
        if len(changed_fields) != len(set(changed_fields)):
            errors.append("root.changed_fields must not contain duplicates")

    expanded_groups: dict[str, dict[str, Any]] = {}
    for group_name in GROUP_NAMES:
        group = payload.get(group_name)
        if not isinstance(group, dict):
            errors.append(f"root.{group_name} must be an object")
            continue
        _require_text(group, "label", errors, location=f"root.{group_name}")
        try:
            expanded_groups[group_name] = expand_group(group)
        except ExperimentValidationError as exc:
            errors.extend(f"root.{group_name}: {error}" for error in exc.errors)

    if len(expanded_groups) == 2 and changed_fields:
        actual_differences = diff_paths(
            _comparison_config(expanded_groups["baseline"]),
            _comparison_config(expanded_groups["candidate"]),
        )
        declared_differences = set(changed_fields)
        undeclared = sorted(actual_differences - declared_differences)
        unchanged = sorted(declared_differences - actual_differences)
        if undeclared:
            errors.append(f"changed_fields is missing actual differences: {undeclared}")
        if unchanged:
            errors.append(f"changed_fields contains paths that do not differ: {unchanged}")

    _validate_decision(payload.get("decision"), errors)
    if errors:
        raise ExperimentValidationError(errors)


def expand_group(group: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    profile = group.get("profile")
    if not isinstance(profile, str) or profile not in SUPPORTED_PROFILES:
        raise ExperimentValidationError(
            [f"profile must be one of {sorted(SUPPORTED_PROFILES)}, got {profile!r}"]
        )

    request_fields = {key: value for key, value in group.items() if key in GROUP_REQUEST_FIELDS}
    unknown_fields = sorted(set(group) - GROUP_REQUEST_FIELDS - {"label"})
    if unknown_fields:
        errors.append(f"unsupported group fields: {unknown_fields}")

    if profile in NAMED_PROFILES:
        advanced_fields = sorted(set(request_fields) - {"profile"})
        if advanced_fields:
            errors.append(
                f"named profile {profile!r} must not include advanced overrides: {advanced_fields}"
            )
        expanded = {"profile": profile, **expand_retrieve_profile(profile)}
    else:
        required = {"top_k", "retrieval_options", "rerank_options", "query_options"}
        missing = sorted(required - set(request_fields))
        if missing:
            errors.append(f"custom profile is missing required fields: {missing}")
        expanded = request_fields

    _validate_expanded_group(expanded, errors)
    if errors:
        raise ExperimentValidationError(errors)
    return expanded


def group_request_payload(group: dict[str, Any]) -> dict[str, Any]:
    """Return the exact request fields to send for one experiment group."""

    return {key: value for key, value in group.items() if key in GROUP_REQUEST_FIELDS}


def expected_effective_config(group: dict[str, Any]) -> dict[str, Any]:
    expanded = expand_group(group)
    retrieval = expanded["retrieval_options"]
    rerank = expanded["rerank_options"]
    query = expanded["query_options"]
    mode = retrieval["mode"]
    rewrite_enabled = query.get("strategy") != "noop" and bool(query.get("enabled"))
    return {
        "profile": expanded["profile"],
        "top_k": expanded["top_k"],
        "retrieval_mode": mode,
        "vector_top_k": retrieval.get("vector_top_k", 0) if mode != "bm25" else 0,
        "bm25_top_k": retrieval.get("bm25_top_k", 0) if mode != "vector" else 0,
        "rrf_k": retrieval.get("rrf_k") if mode == "hybrid" else None,
        "rerank_enabled": bool(rerank["enabled"]),
        "rerank_top_n": rerank.get("top_n"),
        "rewrite_enabled": rewrite_enabled,
        "synonym_enabled": query.get("synonym_enabled", True),
    }


def diff_paths(left: Any, right: Any, prefix: str = "") -> set[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        differences: set[str] = set()
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else key
            if key not in left or key not in right:
                differences.add(path)
            else:
                differences.update(diff_paths(left[key], right[key], path))
        return differences
    return {prefix} if left != right else set()


def _comparison_config(expanded: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in expanded.items() if key != "profile"}


def _validate_expanded_group(group: dict[str, Any], errors: list[str]) -> None:
    top_k = group.get("top_k")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        errors.append("top_k must be a positive integer")

    retrieval = group.get("retrieval_options")
    if not isinstance(retrieval, dict):
        errors.append("retrieval_options must be an object")
    else:
        if retrieval.get("mode") not in {"vector", "bm25", "hybrid"}:
            errors.append("retrieval_options.mode must be vector, bm25, or hybrid")
        for field in ("vector_top_k", "bm25_top_k", "rrf_k"):
            if field in retrieval:
                value = retrieval[field]
                if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                    errors.append(f"retrieval_options.{field} must be a positive integer")

    rerank = group.get("rerank_options")
    if not isinstance(rerank, dict) or not isinstance(rerank.get("enabled"), bool):
        errors.append("rerank_options.enabled must be a boolean")
    elif "top_n" in rerank:
        top_n = rerank["top_n"]
        if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n < 1:
            errors.append("rerank_options.top_n must be a positive integer")

    query = group.get("query_options")
    if not isinstance(query, dict):
        errors.append("query_options must be an object")
    else:
        if not isinstance(query.get("enabled"), bool):
            errors.append("query_options.enabled must be a boolean")
        if query.get("strategy") not in {"noop", "rewrite"}:
            errors.append("query_options.strategy must be noop or rewrite")
        if "synonym_enabled" in query and not isinstance(query["synonym_enabled"], bool):
            errors.append("query_options.synonym_enabled must be a boolean")


def _validate_decision(value: Any, errors: list[str]) -> None:
    if not isinstance(value, dict):
        errors.append("root.decision must be an object")
        return
    primary = value.get("min_primary_improvement")
    if not _is_finite_number(primary) or primary < 0:
        errors.append("root.decision.min_primary_improvement must be a non-negative number")
    max_failed = value.get("max_failed_cases")
    if not isinstance(max_failed, int) or isinstance(max_failed, bool) or max_failed < 0:
        errors.append("root.decision.max_failed_cases must be a non-negative integer")

    guardrails = value.get("quality_guardrails")
    if not isinstance(guardrails, dict) or not guardrails:
        errors.append("root.decision.quality_guardrails must be a non-empty object")
    else:
        for metric, config in guardrails.items():
            if metric not in SUPPORTED_PRIMARY_METRICS:
                errors.append(f"unsupported quality guardrail metric {metric!r}")
            if not isinstance(config, dict) or not _is_finite_number(config.get("min_delta")):
                errors.append(f"quality guardrail {metric!r} must define numeric min_delta")

    cost_limits = value.get("cost_limits")
    if not isinstance(cost_limits, dict):
        errors.append("root.decision.cost_limits must be an object")
    else:
        for field in ("max_p95_increase_ratio", "max_model_calls_increase"):
            number = cost_limits.get(field)
            if not _is_finite_number(number) or number < 0:
                errors.append(f"root.decision.cost_limits.{field} must be non-negative")


def _require_text(
    value: dict[str, Any], field: str, errors: list[str], *, location: str
) -> str:
    raw = value.get(field)
    if not isinstance(raw, str) or not raw.strip():
        errors.append(f"{location}.{field} must be a non-empty string")
        return ""
    return raw.strip()


def _require_positive_int(
    value: dict[str, Any], field: str, errors: list[str], *, location: str
) -> None:
    raw = value.get(field)
    if not isinstance(raw, int) or isinstance(raw, bool) or raw < 1:
        errors.append(f"{location}.{field} must be a positive integer")


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and float(
        "-inf"
    ) < float(value) < float("inf")
