from __future__ import annotations

import base64
import json

from kombu.utils.json import dumps

from scripts.restore_celery_task import (
    INDEXING_TASK_NAME,
    _extract_document_id,
    _parse_unacked_task,
    build_parser,
)


def _message_payload(document_id: str) -> bytes:
    call = [[document_id], {}, {"callbacks": None, "errbacks": None, "chain": None, "chord": None}]
    body = base64.b64encode(json.dumps(call).encode("utf-8")).decode("ascii")
    message = {
        "body": body,
        "content-encoding": "utf-8",
        "content-type": "application/json",
        "headers": {
            "id": "task-id",
            "task": INDEXING_TASK_NAME,
            "argsrepr": repr((document_id,)),
            "kwargsrepr": "{}",
            "retries": 0,
        },
        "properties": {"body_encoding": "base64"},
    }
    return dumps([message, "", "celery"]).encode("utf-8")


def test_extract_document_id_from_indexing_message() -> None:
    payload = json.loads(_message_payload("document-id").decode("utf-8"))
    message = payload[0]

    assert _extract_document_id(message, INDEXING_TASK_NAME) == "document-id"


def test_parse_unacked_task_reads_task_metadata() -> None:
    task = _parse_unacked_task(
        "delivery-tag",
        _message_payload("document-id"),
        1_789_536_251.0,
    )

    assert task.delivery_tag == "delivery-tag"
    assert task.task_id == "task-id"
    assert task.task_name == INDEXING_TASK_NAME
    assert task.document_id == "document-id"
    assert task.routing_key == "celery"
    assert task.retries == 0


def test_restore_parser_requires_explicit_confirmation() -> None:
    args = build_parser().parse_args(
        ["restore", "--delivery-tag", "delivery-tag", "--dry-run"]
    )

    assert args.command == "restore"
    assert args.dry_run is True
    assert args.yes is False
