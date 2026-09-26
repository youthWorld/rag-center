import logging

import pytest

from app.core.logging import log_llm_call, research_retrieval_log_scope
from app.observability import research_observability


@pytest.mark.asyncio
async def test_research_model_error_log_does_not_echo_provider_exception(caplog) -> None:
    async def failed_call():
        raise RuntimeError("secret candidate body and Authorization header")

    with caplog.at_level(logging.INFO):
        with pytest.raises(RuntimeError):
            await log_llm_call(
                failed_call, model="fast-model", prompt={"payload_logging": "disabled"},
                safe_error=True,
            )

    assert "secret candidate body" not in caplog.text
    assert "Authorization header" not in caplog.text
    assert "exception_type=RuntimeError" in caplog.text


def test_research_observability_failure_uses_only_safe_exception_type(monkeypatch, caplog):
    class FailingClient:
        def trace(self, **kwargs):
            del kwargs
            raise RuntimeError("secret question and Authorization header")

        def flush(self):
            pass

    monkeypatch.setattr(
        research_observability, "get_langfuse_client", lambda settings: FailingClient()
    )
    trace = research_observability.ResearchObservability(
        settings=object(), research_id="research", log_id="log", tenant_id="tenant",
        user_id="user", tenant_plan="pro", kb_ids=["kb"],
        index_versions={"kb": "v1"}, query="secret question",
    )
    with caplog.at_level(logging.WARNING), trace:
        assert trace.trace_id is None
    assert "RESEARCH_TRACE_FAILED" in caplog.text
    assert "secret question" not in caplog.text
    assert "Authorization header" not in caplog.text


@pytest.mark.asyncio
async def test_research_retrieval_embedding_error_is_redacted(caplog) -> None:
    async def failed_embedding():
        raise RuntimeError("secret search query and Authorization header")

    with caplog.at_level(logging.INFO), research_retrieval_log_scope():
        with pytest.raises(RuntimeError):
            await log_llm_call(
                failed_embedding, model="embedding-model", prompt="secret search query"
            )
    assert "exception_type=RuntimeError" in caplog.text
    assert "secret search query" not in caplog.text
    assert "Authorization header" not in caplog.text
