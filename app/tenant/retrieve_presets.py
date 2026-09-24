from __future__ import annotations

from copy import deepcopy
from typing import Any

RETRIEVE_PROFILE_PRESETS: dict[str, dict[str, Any]] = {
    "speed": {
        "top_k": 5,
        "retrieval_options": {
            "mode": "vector",
            "vector_top_k": 5,
        },
        "rerank_options": {
            "enabled": False,
            "top_n": 10,
        },
        "query_options": {
            "enabled": False,
            "strategy": "noop",
            "synonym_enabled": True,
        },
    },
    "balanced": {
        "top_k": 10,
        "retrieval_options": {
            "mode": "hybrid",
            "vector_top_k": 10,
            "bm25_top_k": 10,
            "rrf_k": 60,
        },
        "rerank_options": {
            "enabled": False,
            "top_n": 10,
        },
        "query_options": {
            "enabled": False,
            "strategy": "noop",
            "synonym_enabled": True,
        },
    },
    "quality": {
        "top_k": 20,
        "retrieval_options": {
            "mode": "hybrid",
            "vector_top_k": 20,
            "bm25_top_k": 20,
            "rrf_k": 60,
        },
        "rerank_options": {
            "enabled": True,
            "top_n": 10,
        },
        "query_options": {
            "enabled": True,
            "strategy": "rewrite",
            "synonym_enabled": True,
        },
    },
    "custom": {},
}


def expand_retrieve_profile(profile: str) -> dict[str, Any]:
    try:
        return deepcopy(RETRIEVE_PROFILE_PRESETS[profile])
    except KeyError as exc:
        raise ValueError(f"unsupported retrieve profile: {profile}") from exc


def get_retrieve_preset(profile: str) -> dict[str, Any]:
    return expand_retrieve_profile(profile)
