from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import Settings  # noqa: E402

DEFAULT_MAX_SCORE = 3.0
DEFAULT_PAGE_SIZE = 100
MAX_SCORE_PAGES = 1000
DEFAULT_DATASET_PROFILE = "balanced"
FEEDBACK_SCORE_NAME = "user_feedback"


class ExportError(RuntimeError):
    """A user-facing export failure."""


class LangfuseAPIError(ExportError):
    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        super().__init__(message)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export low-score Langfuse retrieval feedback as evaluation cases."
    )
    parser.add_argument(
        "--max-score",
        type=float,
        default=DEFAULT_MAX_SCORE,
        help="Export feedback scores strictly below this value (default: 3).",
    )
    parser.add_argument(
        "--days",
        type=_positive_int,
        help="Only include feedback created in the last N days.",
    )
    parser.add_argument(
        "--kb-id",
        help="Only export cases belonging to this knowledge base.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output dataset JSON path.",
    )
    parser.add_argument(
        "--merge",
        type=Path,
        help="Merge candidates into an existing dataset, deduplicated by question.",
    )
    return parser.parse_args()


def _field(payload: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in payload:
            return payload[name]
    return None


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return str(value).strip() or None


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _serialize_datetime(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _error_text(response: httpx.Response, payload: Any) -> str:
    if isinstance(payload, dict):
        for key in ("message", "error", "msg"):
            message = _as_text(payload.get(key))
            if message:
                return message
    return response.text.strip() or "request failed"


class LangfusePublicAPI:
    """Small synchronous wrapper around the Langfuse Public API."""

    def __init__(
        self,
        client: httpx.Client,
        *,
        host: str,
        page_size: int = DEFAULT_PAGE_SIZE,
    ) -> None:
        self.client = client
        self.base_url = host.rstrip("/")
        self.page_size = page_size

    def _get(self, path: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            response = self.client.get(f"{self.base_url}{path}", params=params)
        except httpx.HTTPError as exception:
            raise ExportError(
                f"Langfuse is unavailable: {str(exception) or type(exception).__name__}"
            ) from exception

        try:
            payload = response.json()
        except ValueError as exception:
            raise LangfuseAPIError(
                response.status_code,
                f"Langfuse returned invalid JSON (HTTP {response.status_code})",
            ) from exception

        if not 200 <= response.status_code < 300:
            raise LangfuseAPIError(
                response.status_code,
                f"Langfuse request failed (HTTP {response.status_code}): "
                f"{_error_text(response, payload)}",
            )
        if not isinstance(payload, dict):
            raise LangfuseAPIError(response.status_code, "Langfuse returned an invalid object")
        return payload

    def list_scores(
        self,
        *,
        max_score: float,
        from_timestamp: datetime | None,
    ) -> list[dict[str, Any]]:
        scores: list[dict[str, Any]] = []
        page = 1
        while True:
            if page > MAX_SCORE_PAGES:
                raise ExportError(
                    f"Langfuse scores pagination exceeded {MAX_SCORE_PAGES} pages"
                )
            params: dict[str, Any] = {
                "page": page,
                "limit": self.page_size,
                "name": FEEDBACK_SCORE_NAME,
                "operator": "<",
                "value": max_score,
            }
            if from_timestamp is not None:
                params["fromTimestamp"] = _serialize_datetime(from_timestamp)

            payload = self._get("/api/public/scores", params=params)
            page_items = payload.get("data")
            if not isinstance(page_items, list):
                raise LangfuseAPIError(200, "Langfuse scores response has no data list")
            scores.extend(item for item in page_items if isinstance(item, dict))

            meta = payload.get("meta")
            total_pages = (
                _as_int(_field(meta, "totalPages", "total_pages"))
                if isinstance(meta, dict)
                else None
            )
            if total_pages is not None and page >= total_pages:
                break
            if not page_items or len(page_items) < self.page_size:
                break
            page += 1

        return scores

    def get_trace(self, trace_id: str) -> dict[str, Any]:
        payload = self._get(f"/api/public/traces/{quote(trace_id, safe='')}")
        trace = payload.get("data")
        if isinstance(trace, dict):
            return trace
        return payload


def _load_settings() -> Settings:
    try:
        settings = Settings()
    except Exception as exception:
        raise ExportError(
            f"failed to load .env settings: {str(exception) or type(exception).__name__}"
        ) from exception

    missing = [
        name
        for name, value in (
            ("LANGFUSE_HOST", settings.langfuse_host),
            ("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key),
            ("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key),
        )
        if not str(value).strip()
    ]
    if missing:
        raise ExportError(
            "Langfuse credentials are incomplete; configure " + ", ".join(missing) + " in .env"
        )
    return settings


def _trace_metadata(trace: dict[str, Any]) -> dict[str, Any]:
    metadata = trace.get("metadata")
    if isinstance(metadata, dict):
        return metadata
    if isinstance(metadata, str) and metadata.strip():
        try:
            decoded = json.loads(metadata)
        except json.JSONDecodeError:
            return {}
        return decoded if isinstance(decoded, dict) else {}
    return {}


def _trace_query(trace: dict[str, Any]) -> str | None:
    trace_input = trace.get("input")
    if isinstance(trace_input, dict):
        return _as_text(_field(trace_input, "query"))
    return _as_text(trace_input)


def _trace_kb_id(trace: dict[str, Any]) -> str | None:
    metadata = _trace_metadata(trace)
    return _as_text(_field(metadata, "kb_id", "kbId"))


def _trace_log_id(trace: dict[str, Any]) -> str | None:
    metadata = _trace_metadata(trace)
    return _as_text(_field(metadata, "log_id", "logId"))


def _score_timestamp(score: dict[str, Any]) -> datetime | None:
    return _parse_datetime(_field(score, "timestamp", "createdAt", "created_at"))


def _build_case(
    score: dict[str, Any],
    trace: dict[str, Any],
    *,
    trace_id: str,
    kb_id: str,
    case_id: str,
) -> dict[str, Any] | None:
    question = _trace_query(trace)
    if question is None:
        return None

    score_value = _as_number(_field(score, "value"))
    if score_value is None:
        return None
    score_output: int | float = int(score_value) if score_value.is_integer() else score_value
    comment = _as_text(_field(score, "comment")) or ""
    source: dict[str, Any] = {
        "trace_id": trace_id,
        "feedback_score": score_output,
        "feedback_comment": comment,
    }
    log_id = _trace_log_id(trace)
    if log_id is not None:
        source["log_id"] = log_id

    return {
        "id": case_id,
        "question": question,
        "ground_truth": "",
        "kb_id": kb_id,
        "source": source,
    }


def collect_cases(
    api: LangfusePublicAPI,
    *,
    max_score: float = DEFAULT_MAX_SCORE,
    days: int | None = None,
    kb_id: str | None = None,
) -> list[dict[str, Any]]:
    if max_score <= 0:
        raise ExportError("--max-score must be greater than zero")

    from_timestamp = datetime.now(UTC) - timedelta(days=days) if days is not None else None
    scores = api.list_scores(max_score=max_score, from_timestamp=from_timestamp)
    cases: list[dict[str, Any]] = []
    used_ids: set[str] = set()

    for score in scores:
        if _as_text(_field(score, "name")) != FEEDBACK_SCORE_NAME:
            continue
        score_value = _as_number(_field(score, "value"))
        if score_value is None or score_value >= max_score:
            continue
        if from_timestamp is not None:
            timestamp = _score_timestamp(score)
            if timestamp is not None and timestamp < from_timestamp:
                continue

        trace_id = _as_text(_field(score, "traceId", "trace_id"))
        if trace_id is None:
            print("warning: skipped feedback without trace_id", file=sys.stderr)
            continue

        try:
            trace = api.get_trace(trace_id)
        except LangfuseAPIError as exception:
            if exception.status_code == 404:
                print(f"warning: trace not found, skipped trace_id={trace_id}", file=sys.stderr)
                continue
            raise

        trace_kb_id = _trace_kb_id(trace)
        if kb_id is not None and trace_kb_id is not None and trace_kb_id != kb_id:
            continue
        effective_kb_id = trace_kb_id or kb_id
        if effective_kb_id is None:
            print(
                f"warning: skipped trace_id={trace_id} because metadata.kb_id is missing",
                file=sys.stderr,
            )
            continue

        base_id = f"lf_{trace_id[:12]}"
        case_id = base_id
        suffix = 2
        while case_id in used_ids:
            case_id = f"{base_id}_{suffix}"
            suffix += 1

        case = _build_case(
            score,
            trace,
            trace_id=trace_id,
            kb_id=effective_kb_id,
            case_id=case_id,
        )
        if case is None:
            print(
                f"warning: skipped trace_id={trace_id} because input.query is missing",
                file=sys.stderr,
            )
            continue
        used_ids.add(case_id)
        cases.append(case)

    return cases


def _validate_dataset(payload: Any, *, path: Path) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ExportError(f"dataset must contain a JSON object: {path}")
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ExportError(f"dataset cases must be a JSON array: {path}")
    return payload


def load_dataset(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exception:
        raise ExportError(f"dataset does not exist: {path}") from exception
    except OSError as exception:
        raise ExportError(f"failed to read dataset {path}: {exception}") from exception
    except json.JSONDecodeError as exception:
        raise ExportError(f"dataset contains invalid JSON: {exception}") from exception
    return _validate_dataset(payload, path=path)


def _question_key(case: dict[str, Any]) -> str:
    value = case.get("question")
    return value.strip() if isinstance(value, str) else ""


def _has_ground_truth(case: dict[str, Any]) -> bool:
    value = case.get("ground_truth")
    return isinstance(value, str) and bool(value.strip())


def merge_cases(existing: list[Any], candidates: list[dict[str, Any]]) -> list[Any]:
    merged = list(existing)
    by_question = {
        _question_key(case): index
        for index, case in enumerate(merged)
        if isinstance(case, dict) and _question_key(case)
    }
    for candidate in candidates:
        question_key = _question_key(candidate)
        if not question_key:
            continue
        existing_index = by_question.get(question_key)
        if existing_index is None:
            merged.append(candidate)
            by_question[question_key] = len(merged) - 1
            continue

        current = merged[existing_index]
        if not isinstance(current, dict) or _has_ground_truth(current):
            continue

        updated = dict(candidate)
        updated["id"] = current.get("id", updated.get("id"))
        updated["ground_truth"] = current.get("ground_truth", "")
        current_source = current.get("source")
        candidate_source = updated.get("source")
        if isinstance(current_source, dict) and isinstance(candidate_source, dict):
            updated["source"] = {**current_source, **candidate_source}
        merged[existing_index] = updated
    return merged


def build_dataset(name: str, *, cases: list[dict[str, Any]], kb_id: str | None) -> dict[str, Any]:
    unique_kb_ids = {case["kb_id"] for case in cases if case.get("kb_id")}
    dataset_kb_id = kb_id or (next(iter(unique_kb_ids)) if len(unique_kb_ids) == 1 else "")
    output_cases: list[dict[str, Any]] = []
    for case in cases:
        output_case = dict(case)
        if dataset_kb_id and output_case.get("kb_id") == dataset_kb_id:
            output_case.pop("kb_id", None)
        output_cases.append(output_case)
    return {
        "name": name,
        "kb_id": dataset_kb_id,
        "default_profile": DEFAULT_DATASET_PROFILE,
        "cases": output_cases,
    }


def write_dataset(path: Path, payload: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as exception:
        raise ExportError(f"failed to write dataset {path}: {exception}") from exception


def _pending_ground_truth_count(cases: list[Any]) -> int:
    return sum(
        1
        for case in cases
        if isinstance(case, dict) and not _has_ground_truth(case)
    )


def main() -> int:
    args = parse_args()
    try:
        settings = _load_settings()
        with httpx.Client(
            auth=(settings.langfuse_public_key, settings.langfuse_secret_key),
            timeout=30.0,
            trust_env=False,
        ) as client:
            api = LangfusePublicAPI(client, host=settings.langfuse_host)
            candidates = collect_cases(
                api,
                max_score=args.max_score,
                days=args.days,
                kb_id=args.kb_id,
            )

        dataset = build_dataset(args.output.stem, cases=candidates, kb_id=args.kb_id)
        if args.merge is not None:
            existing = load_dataset(args.merge)
            dataset = dict(existing)
            dataset["cases"] = merge_cases(existing["cases"], candidates)
            if not dataset.get("kb_id") and args.kb_id:
                dataset["kb_id"] = args.kb_id
            if not dataset.get("name"):
                dataset["name"] = args.output.stem
        write_dataset(args.output, dataset)
    except (ExportError, OSError) as exception:
        print(f"error: {exception}", file=sys.stderr)
        return 1

    print(f"exported candidates: {len(candidates)}")
    print(f"dataset cases: {len(dataset['cases'])}")
    print(f"pending ground_truth: {_pending_ground_truth_count(dataset['cases'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
