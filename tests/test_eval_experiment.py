from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from app.evaluation.experiment import (
    ExperimentValidationError,
    expand_group,
    expected_effective_config,
    load_and_validate_experiment,
    validate_experiment,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUERY_REWRITE_EXPERIMENT = PROJECT_ROOT / "eval" / "experiments" / "query_rewrite.json"
EVIDENCE_EXPERIMENT = PROJECT_ROOT / "eval" / "experiments" / "evidence_orchestration.json"
EXPERIMENT_NAMES = (
    "hybrid_search.json",
    "query_rewrite.json",
    "synonym_expansion.json",
    "profile_speed_vs_balanced.json",
    "profile_balanced_vs_quality.json",
)


@pytest.fixture
def valid_experiment() -> dict:
    return json.loads(QUERY_REWRITE_EXPERIMENT.read_text(encoding="utf-8"))


def test_query_rewrite_experiment_is_valid() -> None:
    experiment = load_and_validate_experiment(QUERY_REWRITE_EXPERIMENT)

    baseline = expected_effective_config(experiment["baseline"])
    candidate = expected_effective_config(experiment["candidate"])

    assert baseline["rewrite_enabled"] is False
    assert candidate["rewrite_enabled"] is True
    assert baseline["synonym_enabled"] is False
    assert baseline["bm25_top_k"] == 0


def test_evidence_experiment_uses_fixed_shared_pipeline() -> None:
    experiment = load_and_validate_experiment(EVIDENCE_EXPERIMENT)

    baseline = expected_effective_config(experiment["baseline"])
    candidate = expected_effective_config(experiment["candidate"])

    assert experiment["changed_fields"] == ["evidence_options.enabled"]
    assert experiment["primary_metric"] == "context_precision"
    assert experiment["concurrency"] == experiment["ragas_concurrency"] == 4
    assert experiment["warmup_cases"] == 0
    assert baseline["top_k"] == candidate["top_k"] == 10
    assert baseline["vector_top_k"] == candidate["vector_top_k"] == 20
    assert baseline["bm25_top_k"] == candidate["bm25_top_k"] == 20
    assert baseline["rerank_enabled"] is candidate["rerank_enabled"] is True
    assert baseline["evidence_enabled"] is False
    assert candidate["evidence_enabled"] is True


def test_evidence_experiment_rejects_different_max_items() -> None:
    experiment = json.loads(EVIDENCE_EXPERIMENT.read_text(encoding="utf-8"))
    experiment["candidate"]["evidence_options"]["max_items"] = 8

    with pytest.raises(ExperimentValidationError, match="same evidence_options.max_items"):
        validate_experiment(experiment)


@pytest.mark.parametrize("filename", EXPERIMENT_NAMES)
def test_all_optimization_fifteen_experiments_are_independently_valid(
    filename: str,
) -> None:
    experiment = load_and_validate_experiment(PROJECT_ROOT / "eval" / "experiments" / filename)

    assert experiment["dataset"] == "eval/datasets/golden_basic_20.json"
    assert experiment["suite"] == "basic"
    assert experiment["warmup_cases"] == 0
    assert experiment["decision"]["max_failed_cases"] == 0


def test_undeclared_group_difference_is_rejected(valid_experiment: dict) -> None:
    experiment = copy.deepcopy(valid_experiment)
    experiment["candidate"]["top_k"] = 6

    with pytest.raises(ExperimentValidationError, match="missing actual differences"):
        validate_experiment(experiment)


def test_changed_field_must_represent_an_actual_difference(valid_experiment: dict) -> None:
    experiment = copy.deepcopy(valid_experiment)
    experiment["changed_fields"].append("rerank_options.enabled")

    with pytest.raises(ExperimentValidationError, match="paths that do not differ"):
        validate_experiment(experiment)


def test_named_profile_rejects_advanced_overrides() -> None:
    with pytest.raises(ExperimentValidationError, match="must not include advanced overrides"):
        expand_group(
            {
                "label": "speed",
                "profile": "speed",
                "top_k": 10,
            }
        )


def test_named_profile_expands_existing_project_preset() -> None:
    expanded = expand_group({"label": "quality", "profile": "quality"})

    assert expanded["retrieval_options"] == {
        "mode": "hybrid",
        "vector_top_k": 20,
        "bm25_top_k": 20,
        "rrf_k": 60,
    }
    assert expanded["rerank_options"] == {"enabled": True, "top_n": 10}
    assert expanded["query_options"] == {
        "enabled": True,
        "strategy": "rewrite",
        "synonym_enabled": True,
    }


def test_invalid_decision_threshold_is_rejected(valid_experiment: dict) -> None:
    experiment = copy.deepcopy(valid_experiment)
    experiment["decision"]["cost_limits"]["max_p95_increase_ratio"] = -1

    with pytest.raises(ExperimentValidationError, match="must be non-negative"):
        validate_experiment(experiment)
