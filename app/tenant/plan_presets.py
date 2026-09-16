from __future__ import annotations

from copy import deepcopy
from typing import Any

PLAN_PRESETS: dict[str, dict[str, Any]] = {
    "free": {
        "features": {
            "allowed_profiles": ["speed"],
            "hybrid_allowed": False,
            "rerank_allowed": False,
            "query_rewrite_allowed": False,
        },
        "limits": {
            "retrieve_qps": 3,
            "retrieve_daily": 500,
            "max_kb": 1,
            "max_documents_per_kb": 30,
            "max_processing_documents": 1,
        },
    },
    "standard": {
        "features": {
            "allowed_profiles": ["speed", "balanced", "custom"],
            "hybrid_allowed": True,
            "rerank_allowed": False,
            "query_rewrite_allowed": False,
        },
        "limits": {
            "retrieve_qps": 10,
            "retrieve_daily": 5_000,
            "max_kb": 5,
            "max_documents_per_kb": 200,
            "max_processing_documents": 2,
        },
    },
    "pro": {
        "features": {
            "allowed_profiles": ["speed", "balanced", "quality", "custom"],
            "hybrid_allowed": True,
            "rerank_allowed": True,
            "query_rewrite_allowed": True,
        },
        "limits": {
            "retrieve_qps": 50,
            "retrieve_daily": 100_000,
            "max_kb": 50,
            "max_documents_per_kb": 5_000,
            "max_processing_documents": 10,
        },
    },
}


def get_plan_preset(plan: str) -> dict[str, Any]:
    normalized_plan = plan.strip().lower()
    try:
        return deepcopy(PLAN_PRESETS[normalized_plan])
    except KeyError as exc:
        raise ValueError(f"unsupported tenant plan: {plan}") from exc
