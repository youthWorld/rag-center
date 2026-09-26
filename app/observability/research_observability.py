from __future__ import annotations

from typing import Any

from app.core.config import Settings
from app.core.logging import get_logger
from app.observability.langfuse_client import get_langfuse_client

logger = get_logger(__name__)


class ResearchObservability:
    """Best-effort Langfuse trace for one controlled research request."""

    def __init__(
        self,
        *,
        settings: Settings,
        research_id: str,
        log_id: str,
        tenant_id: str,
        user_id: str,
        tenant_plan: str,
        kb_ids: list[str],
        index_versions: dict[str, str],
        query: str,
    ) -> None:
        self.settings = settings
        self.research_id = research_id
        self.log_id = log_id
        self.tenant_id = tenant_id
        self.user_id = user_id
        self.tenant_plan = tenant_plan
        self.kb_ids = kb_ids
        self.index_versions = index_versions
        self.query = query
        self.client: Any | None = None
        self.trace: Any | None = None
        self.trace_id: str | None = None

    def __enter__(self) -> ResearchObservability:
        self.client = get_langfuse_client(self.settings)
        if self.client is None:
            return self
        try:
            self.trace = self.client.trace(
                name="rag_research",
                user_id=self.user_id,
                input={
                    "query": self.query,
                    "kb_ids": self.kb_ids,
                    "profile": "research_fixed",
                },
                metadata={
                    "research_id": self.research_id,
                    "log_id": self.log_id,
                    "tenant_id": self.tenant_id,
                    "user_id": self.user_id,
                    "tenant_plan": self.tenant_plan,
                    "profile": "research_fixed",
                    "index_versions": self.index_versions,
                },
            )
            raw_trace_id = getattr(self.trace, "id", None)
            self.trace_id = str(raw_trace_id) if raw_trace_id else None
        except Exception as exception:
            self.trace = None
            self.trace_id = None
            logger.warning(
                "RESEARCH_TRACE_FAILED | error=%s",
                type(exception).__name__,
            )
        return self

    def span(
        self,
        name: str,
        *,
        input: dict[str, Any] | None = None,
        output: dict[str, Any] | None = None,
    ) -> None:
        if self.trace is None:
            return
        try:
            span = self.trace.span(name=name)
            span.end(input=input or {}, output=output or {})
        except Exception as exception:
            logger.warning(
                "RESEARCH_SPAN_FAILED | span=%s | error=%s",
                name,
                type(exception).__name__,
            )

    def finish(self, *, output: dict[str, Any]) -> None:
        if self.trace is None:
            return
        try:
            self.trace.update(output=output)
        except Exception as exception:
            logger.warning(
                "RESEARCH_TRACE_UPDATE_FAILED | error=%s",
                type(exception).__name__,
            )

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        if exc is not None and self.trace is not None:
            try:
                self.trace.update(metadata={"error_type": type(exc).__name__})
            except Exception:
                pass
        if self.client is not None:
            try:
                self.client.flush()
            except Exception as exception:
                logger.warning(
                    "RESEARCH_TRACE_FLUSH_FAILED | error=%s",
                    type(exception).__name__,
                )
        return False
