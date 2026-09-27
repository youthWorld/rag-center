"""Task-local logical model attempts, including failed/cancelled sub-retrievals."""

from contextvars import ContextVar

retrieval_usage: ContextVar[dict[str, int] | None] = ContextVar("retrieval_usage", default=None)
retrieval_requests: ContextVar[dict | None] = ContextVar("retrieval_requests", default=None)


def record_request(kind: str) -> None:
    counts = retrieval_requests.get()
    if counts is not None:
        key = f"{kind}_request_count"
        counts[key] = counts.get(key, 0) + 1
    if kind == "embedding":
        record_retrieval_model_call(kind)


def record_retrieval_model_call(kind: str) -> None:
    counts = retrieval_usage.get()
    if counts is not None:
        counts[kind] = counts.get(kind, 0) + 1
