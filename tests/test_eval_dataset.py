from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from app.evaluation.dataset import (
    DatasetValidationError,
    load_and_validate_dataset,
    select_dataset_cases,
    validate_dataset,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DATASET = PROJECT_ROOT / "eval" / "datasets" / "golden_basic_20.json"


@pytest.fixture
def valid_dataset() -> dict:
    return json.loads(GOLDEN_DATASET.read_text(encoding="utf-8"))


def test_golden_basic_20_is_valid_and_selects_cases_in_suite_order() -> None:
    dataset = load_and_validate_dataset(GOLDEN_DATASET)

    cases = select_dataset_cases(dataset, "basic")

    assert len(cases) == 20
    assert [case["id"] for case in cases] == dataset["suites"]["basic"]["case_ids"]


def test_duplicate_case_id_is_rejected(valid_dataset: dict) -> None:
    dataset = copy.deepcopy(valid_dataset)
    dataset["cases"][1]["id"] = dataset["cases"][0]["id"]

    with pytest.raises(DatasetValidationError, match="duplicate id"):
        validate_dataset(dataset)


def test_empty_ground_truth_is_rejected(valid_dataset: dict) -> None:
    dataset = copy.deepcopy(valid_dataset)
    dataset["cases"][0]["ground_truth"] = "  "

    with pytest.raises(DatasetValidationError, match="ground_truth must be a non-empty string"):
        validate_dataset(dataset)


def test_unknown_kb_alias_is_rejected(valid_dataset: dict) -> None:
    dataset = copy.deepcopy(valid_dataset)
    dataset["cases"][0]["kb_alias"] = "missing"

    with pytest.raises(DatasetValidationError, match="unknown knowledge base 'missing'"):
        validate_dataset(dataset)


def test_suite_reference_to_missing_case_is_rejected(valid_dataset: dict) -> None:
    dataset = copy.deepcopy(valid_dataset)
    dataset["suites"]["basic"]["case_ids"][0] = "missing_case"

    with pytest.raises(DatasetValidationError, match="references unknown case 'missing_case'"):
        validate_dataset(dataset)


def test_invalid_case_type_is_rejected(valid_dataset: dict) -> None:
    dataset = copy.deepcopy(valid_dataset)
    dataset["cases"][0]["case_type"] = "unsupported"

    with pytest.raises(DatasetValidationError, match="case_type must be one of"):
        validate_dataset(dataset)


@pytest.mark.parametrize("field", ["expected_documents", "expected_headings", "tags"])
def test_optional_string_list_fields_reject_invalid_types(
    valid_dataset: dict, field: str
) -> None:
    dataset = copy.deepcopy(valid_dataset)
    dataset["cases"][0][field] = ["valid", 1]

    with pytest.raises(DatasetValidationError, match=rf"{field} must be an array"):
        validate_dataset(dataset)


def test_basic_suite_requires_exact_count_and_distribution(valid_dataset: dict) -> None:
    dataset = copy.deepcopy(valid_dataset)
    dataset["suites"]["basic"]["case_ids"].pop()

    with pytest.raises(DatasetValidationError) as exc_info:
        validate_dataset(dataset)

    assert "must contain exactly 20 cases, got 19" in str(exc_info.value)
    assert "must contain 5 multi_document cases, got 4" in str(exc_info.value)


def test_invalid_dataset_reports_multiple_locations(valid_dataset: dict) -> None:
    dataset = copy.deepcopy(valid_dataset)
    dataset["schema_version"] = ""
    dataset["knowledge_bases"]["ecommerce"]["kb_id"] = ""
    dataset["cases"][0]["question"] = ""

    with pytest.raises(DatasetValidationError) as exc_info:
        validate_dataset(dataset)

    assert len(exc_info.value.errors) >= 3
    assert "root.schema_version" in str(exc_info.value)
    assert "root.knowledge_bases.ecommerce.kb_id" in str(exc_info.value)
    assert "root.cases[0].question" in str(exc_info.value)
