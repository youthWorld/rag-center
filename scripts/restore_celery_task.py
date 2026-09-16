from __future__ import annotations

import argparse
import base64
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kombu import Connection
from kombu.serialization import loads as deserialize_message_body
from kombu.utils.encoding import bytes_to_str
from kombu.utils.json import loads as load_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import settings  # noqa: E402

INDEXING_TASK_NAME = "app.tasks.indexing.index_document_task"


@dataclass(frozen=True)
class UnackedTask:
    delivery_tag: str
    task_id: str | None
    task_name: str | None
    document_id: str | None
    args_repr: str | None
    kwargs_repr: str | None
    retries: int | None
    exchange: str
    routing_key: str
    delivered_at: datetime | None


class BrokerInspector:
    def __init__(self, broker_url: str) -> None:
        self.connection = Connection(broker_url)
        self.channel = self.connection.channel()
        self.client = self.channel.client

    def close(self) -> None:
        self.connection.close()

    def list_tasks(self) -> list[UnackedTask]:
        indexed_tags = self.client.zrange(
            self.channel.unacked_index_key,
            0,
            -1,
            withscores=True,
        )
        indexed_tag_names = {bytes_to_str(tag) for tag, _score in indexed_tags}
        tasks = [
            self._read_task(bytes_to_str(tag), score)
            for tag, score in indexed_tags
        ]

        # Include hash entries that do not have an index entry so the operator
        # can see an inconsistent Broker state instead of missing a message.
        for raw_tag in self.client.hkeys(self.channel.unacked_key):
            tag = bytes_to_str(raw_tag)
            if tag not in indexed_tag_names:
                tasks.append(self._read_task(tag, None))
        return tasks

    def get_task(self, delivery_tag: str) -> UnackedTask | None:
        raw_payload = self.client.hget(self.channel.unacked_key, delivery_tag)
        if raw_payload is None:
            return None
        score = self.client.zscore(self.channel.unacked_index_key, delivery_tag)
        return _parse_unacked_task(delivery_tag, raw_payload, score)

    def restore(self, delivery_tag: str) -> None:
        self.channel.qos.restore_by_tag(delivery_tag)

    def _read_task(self, delivery_tag: str, score: float | None) -> UnackedTask:
        raw_payload = self.client.hget(self.channel.unacked_key, delivery_tag)
        if raw_payload is None:
            return UnackedTask(
                delivery_tag=delivery_tag,
                task_id=None,
                task_name="<missing payload>",
                document_id=None,
                args_repr=None,
                kwargs_repr=None,
                retries=None,
                exchange="",
                routing_key="",
                delivered_at=_timestamp(score),
            )
        return _parse_unacked_task(delivery_tag, raw_payload, score)


def _parse_unacked_task(
    delivery_tag: str,
    raw_payload: bytes,
    score: float | None,
) -> UnackedTask:
    try:
        message, exchange, routing_key = load_json(bytes_to_str(raw_payload))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"delivery tag {delivery_tag} contains an invalid Kombu payload"
        ) from exc

    headers = message.get("headers") or {}
    task_name = headers.get("task")
    task_id = headers.get("id")
    document_id = _extract_document_id(message, task_name)
    return UnackedTask(
        delivery_tag=delivery_tag,
        task_id=_optional_string(task_id),
        task_name=_optional_string(task_name),
        document_id=document_id,
        args_repr=_truncate(headers.get("argsrepr")),
        kwargs_repr=_truncate(headers.get("kwargsrepr")),
        retries=_optional_int(headers.get("retries")),
        exchange=_optional_string(exchange) or "",
        routing_key=_optional_string(routing_key) or "",
        delivered_at=_timestamp(score),
    )


def _extract_document_id(message: dict[str, Any], task_name: str | None) -> str | None:
    if task_name != INDEXING_TASK_NAME:
        return None

    try:
        body = message.get("body")
        properties = message.get("properties") or {}
        if properties.get("body_encoding") == "base64":
            body = base64.b64decode(body)
        elif isinstance(body, str):
            body = body.encode("utf-8")

        content_type = message.get("content-type") or properties.get("content_type")
        content_encoding = message.get("content-encoding") or properties.get(
            "content_encoding"
        )
        call = deserialize_message_body(
            body,
            content_type,
            content_encoding,
            accept={"application/json"},
        )
        args = call[0] if isinstance(call, (list, tuple)) and call else None
        document_id = args[0] if isinstance(args, list) and args else None
        return document_id if isinstance(document_id, str) else None
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) else None


def _truncate(value: Any, limit: int = 240) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if len(text) <= limit else f"{text[: limit - 3]}..."


def _timestamp(score: float | None) -> datetime | None:
    if score is None:
        return None
    return datetime.fromtimestamp(float(score), tz=UTC)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect and restore unacknowledged Celery Redis messages."
    )
    parser.add_argument(
        "--broker-url",
        default=settings.celery_broker_url,
        help="Celery Broker URL; defaults to CELERY_BROKER_URL.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List all unacknowledged Celery messages.")

    for command in ("inspect", "restore"):
        command_parser = subparsers.add_parser(command, help=f"{command.title()} one message.")
        command_parser.add_argument("--delivery-tag", required=True)
        command_parser.add_argument(
            "--expected-task",
            help="Reject the operation if the message task name does not match.",
        )
        command_parser.add_argument(
            "--expected-document-id",
            help="Reject the operation if the indexing document ID does not match.",
        )
        if command == "restore":
            mode = command_parser.add_mutually_exclusive_group()
            mode.add_argument(
                "--dry-run",
                action="store_true",
                help="Show the message and make no changes.",
            )
            mode.add_argument(
                "--yes",
                action="store_true",
                help="Confirm that the exact message may be restored.",
            )

    return parser


def _print_task(task: UnackedTask) -> None:
    print(f"delivery tag: {task.delivery_tag}")
    print(f"task id: {task.task_id or '<unknown>'}")
    print(f"task name: {task.task_name or '<unknown>'}")
    print(f"document id: {task.document_id or '<not an indexing task>'}")
    print(f"args: {task.args_repr or '<unknown>'}")
    print(f"kwargs: {task.kwargs_repr or '<unknown>'}")
    print(f"retries: {task.retries if task.retries is not None else '<unknown>'}")
    print(f"exchange: {task.exchange or '<default>'}")
    print(f"routing key: {task.routing_key or '<unknown>'}")
    print(
        "delivered at: "
        f"{task.delivered_at.isoformat() if task.delivered_at else '<unknown>'}"
    )


def _validate_task(task: UnackedTask, args: argparse.Namespace) -> None:
    if args.expected_task and task.task_name != args.expected_task:
        raise SystemExit(
            f"task name mismatch: expected {args.expected_task}, got {task.task_name}"
        )
    if args.expected_document_id and task.document_id != args.expected_document_id:
        raise SystemExit(
            "document ID mismatch: "
            f"expected {args.expected_document_id}, got {task.document_id}"
        )


def run(args: argparse.Namespace) -> int:
    inspector = BrokerInspector(args.broker_url)
    try:
        if args.command == "list":
            tasks = inspector.list_tasks()
            if not tasks:
                print("no unacknowledged Celery messages")
                return 0
            for index, task in enumerate(tasks):
                if index:
                    print("---")
                _print_task(task)
            return 0

        task = inspector.get_task(args.delivery_tag)
        if task is None:
            raise SystemExit(f"delivery tag not found in unacked: {args.delivery_tag}")
        _validate_task(task, args)
        _print_task(task)

        if args.command == "inspect":
            return 0
        if args.dry_run:
            print("status: dry-run; no changes made")
            return 0
        if not args.yes:
            raise SystemExit("restore requires --yes, or use --dry-run to preview")

        inspector.restore(args.delivery_tag)
        print("status: restore requested")
        print("note: another Worker may consume the requeued message immediately")
        return 0
    finally:
        inspector.close()


def main() -> None:
    args = build_parser().parse_args()
    try:
        raise SystemExit(run(args))
    except SystemExit:
        raise
    except Exception as exc:
        raise SystemExit(f"Celery Broker operation failed: {type(exc).__name__}: {exc}") from exc


if __name__ == "__main__":
    main()
