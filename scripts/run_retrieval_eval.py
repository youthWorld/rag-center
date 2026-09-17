from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import Settings  # noqa: E402

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_PROFILE = "balanced"
EVAL_USER_ID = "eval_runner"
LOW_RECALL_THRESHOLD = 0.5
PROFILE_CHOICES = ("speed", "balanced", "quality", "custom")


class EvalError(RuntimeError):
    """A user-facing evaluation failure."""


class RetrieveAPIError(EvalError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RAGAS retrieval metrics against a golden dataset."
    )
    parser.add_argument("--dataset", type=Path, required=True, help="Golden set JSON path.")
    parser.add_argument("--api-key", required=True, help="RAG Center API Key.")
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"RAG Center service URL (default: {DEFAULT_BASE_URL}).",
    )
    parser.add_argument(
        "--profile",
        choices=PROFILE_CHOICES,
        help="Retrieval profile; defaults to dataset.default_profile or balanced.",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    return parser.parse_args()


def resolve_profile(dataset: dict[str, Any], requested_profile: str | None) -> str:
    profile = requested_profile or _as_text(dataset.get("default_profile")) or DEFAULT_PROFILE
    if profile not in PROFILE_CHOICES:
        raise EvalError(f"unsupported retrieval profile in dataset: {profile}")
    return profile


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exception:
        raise EvalError(f"dataset does not exist: {path}") from exception
    except OSError as exception:
        raise EvalError(f"failed to read dataset {path}: {exception}") from exception
    except json.JSONDecodeError as exception:
        raise EvalError(f"dataset contains invalid JSON: {exception}") from exception


def load_dataset(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise EvalError("dataset must contain a JSON object")
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise EvalError("dataset cases must be a JSON array")
    return payload


def prepare_cases(dataset: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    default_kb_id = _as_text(dataset.get("kb_id"))
    prepared: list[dict[str, Any]] = []
    skipped = 0
    for index, raw_case in enumerate(dataset["cases"], start=1):
        if not isinstance(raw_case, dict):
            raise EvalError(f"case #{index} must be a JSON object")
        case_id = _as_text(raw_case.get("id")) or f"case_{index}"
        ground_truth = _as_text(raw_case.get("ground_truth"))
        if not ground_truth:
            skipped += 1
            continue
        question = _as_text(raw_case.get("question"))
        if not question:
            raise EvalError(f"case {case_id} has no question")
        kb_id = _as_text(raw_case.get("kb_id")) or default_kb_id
        if not kb_id:
            raise EvalError(f"case {case_id} has no kb_id")
        prepared.append(
            {
                "id": case_id,
                "kb_id": kb_id,
                "question": question,
                "ground_truth": ground_truth,
            }
        )
    return prepared, skipped


def _response_error(response: httpx.Response, payload: Any) -> str:
    if isinstance(payload, dict):
        code = payload.get("code")
        message = _as_text(payload.get("msg")) or _as_text(payload.get("message"))
        if code is not None and message:
            return f"code={code}, msg={message}"
        if message:
            return message
    return response.text.strip() or f"HTTP {response.status_code}"


def retrieve_case(
    client: httpx.Client,
    *,
    base_url: str,
    api_key: str,
    case: dict[str, Any],
    profile: str,
) -> list[str]:
    payload = {
        "kb_id": case["kb_id"],
        "user_id": EVAL_USER_ID,
        "query": case["question"],
        "profile": profile,
    }
    try:
        response = client.post(
            f"{base_url.rstrip('/')}/api/v1/rag/retrieve",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
    except httpx.HTTPError as exception:
        raise RetrieveAPIError(
            f"case {case['id']} request failed: "
            f"{str(exception) or type(exception).__name__}"
        ) from exception

    try:
        response_payload = response.json()
    except ValueError as exception:
        raise RetrieveAPIError(
            f"case {case['id']} returned invalid JSON (HTTP {response.status_code})"
        ) from exception

    if not 200 <= response.status_code < 300:
        raise RetrieveAPIError(
            f"case {case['id']} retrieve failed: {_response_error(response, response_payload)}"
        )
    if not isinstance(response_payload, dict):
        raise RetrieveAPIError(f"case {case['id']} returned an invalid response object")
    if response_payload.get("code", 0) != 0:
        raise RetrieveAPIError(
            f"case {case['id']} retrieve failed: {_response_error(response, response_payload)}"
        )

    data = response_payload.get("data")
    if not isinstance(data, dict):
        raise RetrieveAPIError(f"case {case['id']} response has no data object")
    chunks = data.get("retrieved_chunks")
    if not isinstance(chunks, list):
        raise RetrieveAPIError(f"case {case['id']} response has no retrieved_chunks list")

    contexts: list[str] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        content = _as_text(chunk.get("content"))
        if content:
            contexts.append(content)
    return contexts


def _load_eval_settings() -> Settings:
    try:
        settings = Settings()
    except Exception as exception:
        raise EvalError(
            f"failed to load .env settings: {str(exception) or type(exception).__name__}"
        ) from exception
    if not settings.model_api_key and not settings.llm_api_key:
        raise EvalError("MODEL_API_KEY or LLM_API_KEY is required for RAGAS evaluation")

    # RAGAS' default OpenAI-compatible adapter reads these conventional names.
    if settings.model_api_key:
        api_key = settings.model_api_key
        base_url = settings.model_base_url
    else:
        api_key = settings.llm_api_key
        base_url = settings.llm_base_url
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_BASE_URL"] = base_url
    os.environ["OPENAI_API_BASE"] = base_url
    return settings


def _build_ragas_llm(settings: Settings) -> Any:
    try:
        from openai import OpenAI
        from ragas.llms import llm_factory
    except ImportError as exception:
        raise EvalError(
            'RAGAS dependencies are missing; install them with `uv sync --extra eval`'
        ) from exception

    if settings.model_api_key:
        api_key = settings.model_api_key
        base_url = settings.model_base_url
    else:
        api_key = settings.llm_api_key
        base_url = settings.llm_base_url
    model = (
        os.getenv("RAGAS_LLM_MODEL")
        or os.getenv("EVAL_MODEL")
        or settings.llm_model
        or settings.embedding_model
    )
    if "client" in inspect.signature(llm_factory).parameters:
        client = OpenAI(api_key=api_key, base_url=base_url)
        return llm_factory(model, client=client)
    return llm_factory(model=model, base_url=base_url)


def _result_records(result: Any) -> list[dict[str, Any]]:
    if isinstance(result, dict):
        return [result]

    scores = getattr(result, "scores", None)
    if isinstance(scores, list) and all(isinstance(item, dict) for item in scores):
        return list(scores)

    to_pandas = getattr(result, "to_pandas", None)
    if callable(to_pandas):
        frame = to_pandas()
        to_dict = getattr(frame, "to_dict", None)
        if callable(to_dict):
            records = to_dict(orient="records")
            if isinstance(records, list) and all(isinstance(item, dict) for item in records):
                return records

    raise EvalError("RAGAS returned no per-case score records")


def _metric_value(record: dict[str, Any], metric_name: str) -> float | None:
    value = record.get(metric_name)
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(numeric) else numeric


def evaluate_with_ragas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    try:
        from datasets import Dataset
        from ragas import evaluate
        try:
            from ragas.metrics.collections.context_precision import ContextPrecision
            from ragas.metrics.collections.context_recall import ContextRecall
        except ImportError:
            from ragas.metrics import context_precision, context_recall
            modern_metrics = None
        else:
            modern_metrics = (ContextPrecision, ContextRecall)
    except ImportError as exception:
        raise EvalError(
            'RAGAS dependencies are missing; install them with `uv sync --extra eval`'
        ) from exception

    dataset = Dataset.from_list(
        [
            {
                "user_input": row["question"],
                "retrieved_contexts": row["contexts"],
                "reference": row["ground_truth"],
            }
            for row in rows
        ]
    )
    try:
        settings = _load_eval_settings()
        evaluator_llm = _build_ragas_llm(settings)
        if modern_metrics is None:
            metrics = [context_precision, context_recall]
        else:
            context_precision_class, context_recall_class = modern_metrics
            metrics = [
                context_precision_class(evaluator_llm),
                context_recall_class(evaluator_llm),
            ]
        result = evaluate(
            dataset,
            metrics=metrics,
            llm=evaluator_llm,
            raise_exceptions=False,
            show_progress=False,
        )
    except Exception as exception:
        raise EvalError(
            f"RAGAS evaluation failed: {str(exception) or type(exception).__name__}"
        ) from exception

    records = _result_records(result)
    if len(records) != len(rows):
        raise EvalError(
            f"RAGAS returned {len(records)} score rows for {len(rows)} evaluation cases"
        )
    scored: list[dict[str, Any]] = []
    for row, record in zip(rows, records, strict=True):
        scored.append(
            {
                "id": row["id"],
                "question": row["question"],
                "context_precision": _metric_value(record, "context_precision"),
                "context_recall": _metric_value(record, "context_recall"),
            }
        )
    return scored


def _average(values: list[float | None]) -> float | None:
    available = [value for value in values if value is not None]
    return sum(available) / len(available) if available else None


def build_report(
    dataset: dict[str, Any],
    *,
    dataset_path: Path,
    profile: str,
    scored_cases: list[dict[str, Any]],
    skipped: int,
) -> dict[str, Any]:
    low_recall_cases = [
        case
        for case in scored_cases
        if case["context_recall"] is not None
        and case["context_recall"] <= LOW_RECALL_THRESHOLD
    ]
    return {
        "name": dataset.get("name") or dataset_path.stem,
        "dataset": str(dataset_path),
        "kb_id": dataset.get("kb_id", ""),
        "profile": profile,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "summary": {
            "evaluated": len(scored_cases),
            "skipped": skipped,
            "context_precision": _average(
                [case["context_precision"] for case in scored_cases]
            ),
            "context_recall": _average(
                [case["context_recall"] for case in scored_cases]
            ),
        },
        "cases": scored_cases,
        "low_context_recall": low_recall_cases,
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as exception:
        raise EvalError(f"failed to write report {path}: {exception}") from exception


def print_report(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print(
        "summary: "
        f"evaluated={summary['evaluated']} "
        f"skipped={summary['skipped']} "
        f"context_precision={summary['context_precision']} "
        f"context_recall={summary['context_recall']}"
    )
    if summary["skipped"]:
        print(f"skipped {summary['skipped']} cases without ground_truth")
    for case in report["cases"]:
        print(
            f"{case['id']}: "
            f"context_precision={case['context_precision']} "
            f"context_recall={case['context_recall']}"
        )
    low_recall = report["low_context_recall"]
    if low_recall:
        print(f"low context_recall cases (<= {LOW_RECALL_THRESHOLD}):")
        for case in low_recall:
            print(f"{case['id']}: context_recall={case['context_recall']}")


def main() -> int:
    args = parse_args()
    try:
        dataset = load_dataset(args.dataset)
        cases, skipped = prepare_cases(dataset)
        profile = resolve_profile(dataset, args.profile)
        rows: list[dict[str, Any]] = []
        with httpx.Client(timeout=60.0, trust_env=False) as client:
            for case in cases:
                contexts = retrieve_case(
                    client,
                    base_url=args.base_url,
                    api_key=args.api_key,
                    case=case,
                    profile=profile,
                )
                rows.append({**case, "contexts": contexts})

        scored_cases = evaluate_with_ragas(rows) if rows else []
        report = build_report(
            dataset,
            dataset_path=args.dataset,
            profile=profile,
            scored_cases=scored_cases,
            skipped=skipped,
        )
        if args.output is not None:
            write_report(args.output, report)
    except EvalError as exception:
        print(f"error: {exception}", file=sys.stderr)
        return 1

    print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
