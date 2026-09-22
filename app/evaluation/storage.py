from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SENSITIVE_KEY_PATTERN = re.compile(
    r"(?:api[_-]?key|authorization|password|secret|access[_-]?token|refresh[_-]?token)",
    re.IGNORECASE,
)


def make_run_id(experiment_id: str, *, suffix: str | None = None) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    parts = [timestamp, experiment_id]
    if suffix:
        safe_suffix = re.sub(r"[^a-zA-Z0-9_-]+", "-", suffix).strip("-")
        if safe_suffix:
            parts.append(safe_suffix)
    return "_".join(parts)


def sanitize_snapshot(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if SENSITIVE_KEY_PATTERN.search(str(key)) else sanitize_snapshot(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [sanitize_snapshot(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(sanitize_snapshot(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(sanitize_snapshot(row), ensure_ascii=False) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise ValueError(f"required result file does not exist: {path}") from exc
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL in {path} at line {index}: {exc.msg}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"invalid JSONL object in {path} at line {index}")
        rows.append(row)
    return rows
