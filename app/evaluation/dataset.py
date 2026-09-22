from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

CASE_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
ALLOWED_CASE_TYPES = {"direct_fact", "paraphrase", "multi_document"}
BASIC_CASE_TYPE_COUNTS = {
    "direct_fact": 10,
    "paraphrase": 5,
    "multi_document": 5,
}
OPTIONAL_STRING_LIST_FIELDS = ("expected_documents", "expected_headings", "tags")


class DatasetValidationError(ValueError):
    """Raised when a Golden Set does not satisfy the unified schema."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("dataset validation failed:\n" + "\n".join(f"- {e}" for e in errors))


def load_dataset(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DatasetValidationError([f"dataset does not exist: {path}"]) from exc
    except OSError as exc:
        raise DatasetValidationError([f"failed to read dataset {path}: {exc}"]) from exc
    except json.JSONDecodeError as exc:
        raise DatasetValidationError(
            [f"dataset contains invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"]
        ) from exc

    if not isinstance(payload, dict):
        raise DatasetValidationError(["root must be a JSON object"])
    return payload


def load_and_validate_dataset(path: Path) -> dict[str, Any]:
    payload = load_dataset(path)
    validate_dataset(payload)
    return payload


def validate_dataset(payload: dict[str, Any]) -> None:
    errors: list[str] = []

    for field in ("schema_version", "name", "version"):
        _require_text(payload, field, errors, location="root")

    knowledge_bases = payload.get("knowledge_bases")
    if not isinstance(knowledge_bases, dict) or not knowledge_bases:
        errors.append("root.knowledge_bases must be a non-empty object")
        knowledge_bases = {}
    else:
        for alias, config in knowledge_bases.items():
            location = f"root.knowledge_bases.{alias}"
            if not isinstance(alias, str) or not alias.strip():
                errors.append("root.knowledge_bases contains a blank alias")
                continue
            if not isinstance(config, dict):
                errors.append(f"{location} must be an object")
                continue
            _require_text(config, "kb_id", errors, location=location)
            _require_text(config, "corpus_version", errors, location=location)

    cases = payload.get("cases")
    if not isinstance(cases, list):
        errors.append("root.cases must be an array")
        cases = []

    case_ids: list[str] = []
    cases_by_id: dict[str, dict[str, Any]] = {}
    for index, case in enumerate(cases):
        location = f"root.cases[{index}]"
        if not isinstance(case, dict):
            errors.append(f"{location} must be an object")
            continue

        case_id = _require_text(case, "id", errors, location=location)
        if case_id:
            case_ids.append(case_id)
            cases_by_id.setdefault(case_id, case)
            if not CASE_ID_PATTERN.fullmatch(case_id):
                errors.append(
                    f"{location}.id must contain only lowercase letters, digits, and underscores"
                )

        _require_text(case, "question", errors, location=location)
        _require_text(case, "ground_truth", errors, location=location)
        _require_text(case, "suite", errors, location=location)

        case_type = _require_text(case, "case_type", errors, location=location)
        if case_type and case_type not in ALLOWED_CASE_TYPES:
            errors.append(
                f"{location}.case_type must be one of {sorted(ALLOWED_CASE_TYPES)}, "
                f"got {case_type!r}"
            )

        kb_alias = _require_text(case, "kb_alias", errors, location=location)
        if kb_alias and kb_alias not in knowledge_bases:
            errors.append(f"{location}.kb_alias references unknown knowledge base {kb_alias!r}")

        for field in OPTIONAL_STRING_LIST_FIELDS:
            if field in case:
                _validate_string_list(case[field], errors, location=f"{location}.{field}")

    duplicate_ids = sorted(case_id for case_id, count in Counter(case_ids).items() if count > 1)
    for case_id in duplicate_ids:
        errors.append(f"root.cases contains duplicate id {case_id!r}")

    suites = payload.get("suites")
    if not isinstance(suites, dict) or not suites:
        errors.append("root.suites must be a non-empty object")
        suites = {}
    else:
        for suite_name, suite in suites.items():
            location = f"root.suites.{suite_name}"
            if not isinstance(suite, dict):
                errors.append(f"{location} must be an object")
                continue
            case_refs = suite.get("case_ids")
            if case_refs == "all":
                continue
            if not isinstance(case_refs, list):
                errors.append(f"{location}.case_ids must be an array or the string 'all'")
                continue
            _validate_string_list(case_refs, errors, location=f"{location}.case_ids")
            for case_id in case_refs:
                if isinstance(case_id, str) and case_id not in cases_by_id:
                    errors.append(f"{location}.case_ids references unknown case {case_id!r}")

    basic_suite = suites.get("basic") if isinstance(suites, dict) else None
    basic_case_ids: list[str] = []
    if not isinstance(basic_suite, dict):
        errors.append("root.suites.basic must be an object")
    else:
        raw_basic_case_ids = basic_suite.get("case_ids")
        if isinstance(raw_basic_case_ids, list):
            basic_case_ids = [case_id for case_id in raw_basic_case_ids if isinstance(case_id, str)]
            if len(basic_case_ids) != 20:
                errors.append(
                    "root.suites.basic.case_ids must contain exactly 20 cases, "
                    f"got {len(basic_case_ids)}"
                )
        else:
            errors.append("root.suites.basic.case_ids must be an explicit array of 20 case ids")

    basic_cases = [cases_by_id[case_id] for case_id in basic_case_ids if case_id in cases_by_id]
    basic_type_counts = Counter(
        case.get("case_type") for case in basic_cases if isinstance(case.get("case_type"), str)
    )
    for case_type, expected_count in BASIC_CASE_TYPE_COUNTS.items():
        actual_count = basic_type_counts.get(case_type, 0)
        if actual_count != expected_count:
            errors.append(
                f"basic suite must contain {expected_count} {case_type} cases, got {actual_count}"
            )

    for case_id, case in cases_by_id.items():
        if case.get("suite") == "basic" and case_id not in basic_case_ids:
            errors.append(
                f"root.cases case {case_id!r} declares suite 'basic' but is not referenced by it"
            )

    if errors:
        raise DatasetValidationError(errors)


def select_dataset_cases(payload: dict[str, Any], suite_name: str) -> list[dict[str, Any]]:
    """Return cases in suite order after the dataset has been validated."""

    suites = payload["suites"]
    try:
        suite = suites[suite_name]
    except KeyError as exc:
        raise DatasetValidationError([f"unknown suite {suite_name!r}"]) from exc

    cases = payload["cases"]
    case_ids = suite["case_ids"]
    if case_ids == "all":
        return list(cases)
    cases_by_id = {case["id"]: case for case in cases}
    return [cases_by_id[case_id] for case_id in case_ids]


def _require_text(
    value: dict[str, Any],
    field: str,
    errors: list[str],
    *,
    location: str,
) -> str:
    raw = value.get(field)
    if not isinstance(raw, str) or not raw.strip():
        errors.append(f"{location}.{field} must be a non-empty string")
        return ""
    return raw.strip()


def _validate_string_list(value: Any, errors: list[str], *, location: str) -> None:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        errors.append(f"{location} must be an array of non-empty strings")
