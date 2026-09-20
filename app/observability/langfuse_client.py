from __future__ import annotations

from typing import Any

import httpx
from langfuse import Langfuse

from app.core.config import Settings, get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_langfuse_client: Langfuse | None = None
_langfuse_config: tuple[str, str, str] | None = None


def get_langfuse_client(app_settings: Settings | None = None) -> Langfuse | None:
    """Return the configured shared Langfuse client, or None when unavailable."""

    global _langfuse_client, _langfuse_config

    app_settings = app_settings or get_settings()
    if not app_settings.langfuse_enabled:
        return None
    if not app_settings.langfuse_public_key or not app_settings.langfuse_secret_key:
        logger.warning("LANGFUSE_UNAVAILABLE | reason=missing credentials")
        return None

    config = (
        app_settings.langfuse_host,
        app_settings.langfuse_public_key,
        app_settings.langfuse_secret_key,
    )
    if _langfuse_client is not None and _langfuse_config == config:
        return _langfuse_client

    try:
        _langfuse_client = Langfuse(
            host=app_settings.langfuse_host,
            public_key=app_settings.langfuse_public_key,
            secret_key=app_settings.langfuse_secret_key,
            httpx_client=httpx.Client(trust_env=False),
            enabled=True,
        )
        _langfuse_config = config
    except Exception as exception:
        _langfuse_client = None
        _langfuse_config = None
        logger.warning(
            "LANGFUSE_UNAVAILABLE | reason=client_initialization_failed | error=%s",
            str(exception) or type(exception).__name__,
        )
        return None
    return _langfuse_client


def get_langfuse(app_settings: Settings | None = None) -> Langfuse | None:
    return get_langfuse_client(app_settings)


def has_trace_score(client: Langfuse, *, trace_id: str, name: str) -> bool:
    response = client.fetch_trace(trace_id)
    trace = getattr(response, "data", response)
    return any(getattr(score, "name", None) == name for score in getattr(trace, "scores", ()))


class RetrieveObservability:
    """Best-effort trace lifecycle for one retrieval request."""

    def __init__(
        self,
        *,
        settings: Settings,
        tenant_id: str,
        kb_id: str,
        kb_ids: list[str] | None = None,
        user_id: str,
        profile: str,
        plan: str,
        raw_query: str,
    ) -> None:
        self.settings = settings
        self.tenant_id = tenant_id
        self.kb_id = kb_id
        self.kb_ids = list(kb_ids) if kb_ids is not None else [kb_id]
        self.user_id = user_id
        self.profile = profile
        self.plan = plan
        self.raw_query = raw_query
        self.client: Langfuse | None = None
        self.trace: Any | None = None
        self.trace_id: str | None = None

    def __enter__(self) -> RetrieveObservability:
        self.client = get_langfuse_client(self.settings)
        if self.client is None:
            return self
        try:
            metadata = {
                "tenant_id": self.tenant_id,
                "kb_id": self.kb_id,
                "user_id": self.user_id,
                "profile": self.profile,
                "plan": self.plan,
            }
            if len(self.kb_ids) > 1:
                metadata["kb_ids"] = self.kb_ids
            self.trace = self.client.trace(
                name="rag_retrieve",
                user_id=self.user_id,
                input={"query": self.raw_query},
                metadata=metadata,
            )
            trace_id = getattr(self.trace, "id", None)
            self.trace_id = str(trace_id) if trace_id else None
        except Exception as exception:
            self.trace = None
            self.trace_id = None
            logger.warning(
                "LANGFUSE_TRACE_FAILED | error=%s",
                str(exception) or type(exception).__name__,
            )
        return self

    def record_query_processing(
        self,
        *,
        effective_query: str,
        search_query: str,
        rewrite_latency_ms: int,
        synonym_applied: bool,
        synonym_expansions: list[str],
        degraded: bool,
        degraded_reason: str | None,
    ) -> None:
        self._record_span(
            "query_processing",
            input={"raw_query": self.raw_query},
            output={
                "effective_query": effective_query,
                "search_query": search_query,
                "rewrite_latency_ms": rewrite_latency_ms,
                "synonym_applied": synonym_applied,
                "synonym_expansions": synonym_expansions,
                "degraded": degraded,
                "degraded_reason": degraded_reason,
            },
        )

    def record_retrieval(
        self,
        *,
        search_query: str,
        mode: str,
        vector_count: int,
        bm25_count: int,
        fused_count: int,
        degraded: bool,
        degraded_reason: str | None,
    ) -> None:
        self._record_span(
            "retrieval",
            input={"query": search_query, "mode": mode},
            output={
                "vector_count": vector_count,
                "bm25_count": bm25_count,
                "fused_count": fused_count,
                "degraded": degraded,
                "degraded_reason": degraded_reason,
            },
        )

    def record_rerank(
        self,
        *,
        enabled: bool,
        candidate_count: int,
        degraded: bool,
        error: str | None,
    ) -> None:
        self._record_span(
            "rerank",
            input={"enabled": enabled, "candidate_count": candidate_count},
            output={"degraded": degraded, "error": error},
        )

    def finish(self, *, log_id: str, chunks: list[dict[str, Any]]) -> None:
        if self.trace is None:
            return
        self._safe_trace_update(
            output={
                "chunks": [
                    {"chunk_id": chunk["chunk_id"], "score": chunk["score"]}
                    for chunk in chunks
                ]
            },
            metadata={"log_id": log_id},
        )

    def __exit__(self, exception_type: Any, exception: Any, traceback: Any) -> bool:
        if exception is not None and self.trace is not None:
            self._safe_trace_update(
                metadata={
                    "error": str(exception) or type(exception).__name__,
                }
            )
        self._flush()
        return False

    def _record_span(
        self,
        name: str,
        *,
        input: dict[str, Any],
        output: dict[str, Any],
    ) -> None:
        if self.trace is None:
            return
        try:
            span = self.trace.span(name=name)
            span.end(input=input, output=output)
        except Exception as exception:
            logger.warning(
                "LANGFUSE_SPAN_FAILED | span=%s | error=%s",
                name,
                str(exception) or type(exception).__name__,
            )

    def _safe_trace_update(self, **kwargs: Any) -> None:
        try:
            self.trace.update(**kwargs)
        except Exception as exception:
            logger.warning(
                "LANGFUSE_TRACE_UPDATE_FAILED | error=%s",
                str(exception) or type(exception).__name__,
            )

    def _flush(self) -> None:
        if self.client is None:
            return
        try:
            self.client.flush()
        except Exception as exception:
            logger.warning(
                "LANGFUSE_FLUSH_FAILED | error=%s",
                str(exception) or type(exception).__name__,
            )
