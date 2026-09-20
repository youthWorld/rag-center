import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from app.core.config import Settings
from scripts import run_retrieval_eval
from scripts.export_eval_cases_from_langfuse import (
    LangfusePublicAPI,
    build_dataset,
    collect_cases,
    merge_cases,
)
from scripts.run_retrieval_eval import (
    RetrieveAPIError,
    build_report,
    prepare_cases,
    resolve_profile,
    retrieve_case,
)


class FakeLangfuseAPI:
    def __init__(self) -> None:
        self.trace_calls: list[str] = []

    def list_scores(self, *, max_score: float, from_timestamp):
        assert max_score == 3
        assert from_timestamp is None
        return [
            {
                "name": "user_feedback",
                "value": 2,
                "traceId": "trace-low",
                "comment": "结果不相关",
            },
            {
                "name": "user_feedback",
                "value": 3,
                "traceId": "trace-at-threshold",
            },
            {
                "name": "other_score",
                "value": 1,
                "traceId": "trace-other",
            },
        ]

    def get_trace(self, trace_id: str):
        self.trace_calls.append(trace_id)
        return {
            "id": trace_id,
            "input": {"query": "退款审核需要多长时间？"},
            "metadata": {"kb_id": "kb-a", "log_id": "log-a"},
        }


def test_collect_cases_exports_only_low_feedback_and_trace_metadata() -> None:
    api = FakeLangfuseAPI()

    cases = collect_cases(api, max_score=3)

    assert api.trace_calls == ["trace-low"]
    assert cases == [
        {
            "id": "lf_trace-low",
            "question": "退款审核需要多长时间？",
            "ground_truth": "",
            "kb_ids": ["kb-a"],
            "source": {
                "trace_id": "trace-low",
                "feedback_score": 2,
                "feedback_comment": "结果不相关",
                "log_id": "log-a",
            },
        }
    ]


def test_collect_cases_prefers_multi_kb_metadata_and_filters_by_membership() -> None:
    class MultiKBAPI:
        def list_scores(self, *, max_score: float, from_timestamp):
            assert max_score == 3
            assert from_timestamp is None
            return [
                {
                    "name": "user_feedback",
                    "value": 2,
                    "traceId": "trace-multi",
                },
                {
                    "name": "user_feedback",
                    "value": 2,
                    "traceId": "trace-other",
                },
            ]

        def get_trace(self, trace_id: str):
            if trace_id == "trace-multi":
                return {
                    "input": {"query": "跨库问题"},
                    "metadata": '{"kbIds": ["kb-a", "kb-b"]}',
                }
            return {
                "input": {"query": "单库问题"},
                "metadata": {"kb_id": "kb-c"},
            }

    cases = collect_cases(MultiKBAPI(), max_score=3, kb_id="kb-b")

    assert cases == [
        {
            "id": "lf_trace-multi",
            "question": "跨库问题",
            "ground_truth": "",
            "kb_ids": ["kb-a", "kb-b"],
            "source": {
                "trace_id": "trace-multi",
                "feedback_score": 2,
                "feedback_comment": "",
            },
        }
    ]


def test_langfuse_public_api_uses_basic_auth_and_paginates() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/scores"):
            page = request.url.params.get("page")
            if page == "1":
                return httpx.Response(
                    200,
                    json={
                        "data": [{"name": "user_feedback", "value": 1}],
                        "meta": {"totalPages": 2},
                    },
                    request=request,
                )
            return httpx.Response(
                200,
                json={
                    "data": [{"name": "user_feedback", "value": 2}],
                    "meta": {"totalPages": 2},
                },
                request=request,
            )
        return httpx.Response(200, json={"id": "trace-a"}, request=request)

    with httpx.Client(
        auth=("public", "secret"),
        transport=httpx.MockTransport(handler),
        trust_env=False,
    ) as client:
        api = LangfusePublicAPI(client, host="http://langfuse.local", page_size=1)
        scores = api.list_scores(max_score=3, from_timestamp=None)

    assert len(scores) == 2
    assert [request.url.path for request in requests] == [
        "/api/public/scores",
        "/api/public/scores",
    ]
    assert requests[0].url.params["name"] == "user_feedback"
    assert requests[0].url.params["operator"] == "<"
    assert requests[0].url.params["value"] == "3"
    assert requests[0].headers["Authorization"].startswith("Basic ")


def test_merge_cases_deduplicates_question_without_overwriting_ground_truth() -> None:
    existing = [
        {
            "id": "manual-1",
            "question": "重复问题",
            "ground_truth": "人工确认答案",
            "source": {"origin": "manual"},
        },
        {
            "id": "pending-1",
            "question": "待审核问题",
            "ground_truth": "",
            "source": {"origin": "old"},
        },
    ]
    candidates = [
        {
            "id": "feedback-1",
            "question": "重复问题",
            "ground_truth": "",
            "source": {"trace_id": "trace-1"},
        },
        {
            "id": "feedback-2",
            "question": "待审核问题",
            "ground_truth": "",
            "source": {"trace_id": "trace-2"},
        },
        {
            "id": "feedback-3",
            "question": "新问题",
            "ground_truth": "",
            "source": {"trace_id": "trace-3"},
        },
    ]

    merged = merge_cases(existing, candidates)

    assert len(merged) == 3
    assert merged[0]["ground_truth"] == "人工确认答案"
    assert merged[0]["source"] == {"origin": "manual"}
    assert merged[1]["source"]["trace_id"] == "trace-2"
    assert merged[2]["question"] == "新问题"


def test_retrieve_case_sends_expected_contract_and_returns_contexts() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "code": 0,
                "msg": "success",
                "data": {
                    "retrieved_chunks": [
                        {"content": "第一段"},
                        {"content": "  "},
                        {"content": "第二段"},
                    ]
                },
            },
            request=request,
        )

    case = {
        "id": "case-1",
        "kb_id": "kb-a",
        "question": "测试问题",
        "ground_truth": "标准答案",
    }
    with httpx.Client(transport=httpx.MockTransport(handler), trust_env=False) as client:
        contexts = retrieve_case(
            client,
            base_url="http://rag.local/",
            api_key="rk_live_test",
            case=case,
            profile="balanced",
        )

    assert contexts == ["第一段", "第二段"]
    assert requests[0].headers["Authorization"] == "Bearer rk_live_test"
    assert json.loads(requests[0].content) == {
        "kb_id": "kb-a",
        "user_id": "eval_runner",
        "query": "测试问题",
        "profile": "balanced",
    }


def test_retrieve_case_sends_multi_kb_contract() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"code": 0, "data": {"retrieved_chunks": []}},
            request=request,
        )

    case = {
        "id": "case-multi",
        "kb_ids": ["kb-a", "kb-b"],
        "question": "跨库问题",
        "ground_truth": "标准答案",
    }
    with httpx.Client(transport=httpx.MockTransport(handler), trust_env=False) as client:
        retrieve_case(
            client,
            base_url="http://rag.local",
            api_key="rk_live_test",
            case=case,
            profile="balanced",
        )

    assert json.loads(requests[0].content) == {
        "kb_ids": ["kb-a", "kb-b"],
        "user_id": "eval_runner",
        "query": "跨库问题",
        "profile": "balanced",
    }


def test_retrieve_case_reports_api_errors_without_raw_traceback() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404,
            json={"code": 30001, "msg": "knowledge base not found", "data": {}},
            request=request,
        )

    case = {
        "id": "case-404",
        "kb_id": "kb-missing",
        "question": "测试问题",
        "ground_truth": "标准答案",
    }
    with httpx.Client(transport=httpx.MockTransport(handler), trust_env=False) as client:
        with pytest.raises(RetrieveAPIError, match="code=30001, msg=knowledge base not found"):
            retrieve_case(
                client,
                base_url="http://rag.local",
                api_key="rk_live_test",
                case=case,
                profile="balanced",
            )


def test_prepare_cases_skips_empty_ground_truth() -> None:
    cases, skipped = prepare_cases(
        {
            "kb_id": "kb-a",
            "cases": [
                {"id": "pending", "question": "待补答案", "ground_truth": ""},
                {"id": "ready", "question": "已补答案", "ground_truth": "答案"},
            ],
        }
    )

    assert skipped == 1
    assert cases == [
        {
            "id": "ready",
            "kb_ids": ["kb-a"],
            "question": "已补答案",
            "ground_truth": "答案",
        }
    ]


def test_resolve_profile_prefers_cli_then_dataset_then_balanced() -> None:
    assert resolve_profile({"default_profile": "quality"}, None) == "quality"
    assert resolve_profile({"default_profile": "quality"}, "speed") == "speed"
    assert resolve_profile({}, None) == "balanced"


def test_evaluate_with_ragas_uses_canonical_columns_and_two_metrics(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeDataset:
        @classmethod
        def from_list(cls, rows):
            captured["rows"] = rows
            return "dataset"

    def fake_evaluate(dataset, **kwargs):
        captured["dataset"] = dataset
        captured["kwargs"] = kwargs
        return SimpleNamespace(
            scores=[
                {"context_precision": 0.8, "context_recall": 0.4},
                {"context_precision": 1.0, "context_recall": 0.9},
            ]
        )

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(Dataset=FakeDataset))
    monkeypatch.setitem(sys.modules, "ragas", SimpleNamespace(evaluate=fake_evaluate))
    monkeypatch.setitem(
        sys.modules,
        "ragas.metrics",
        SimpleNamespace(context_precision="precision", context_recall="recall"),
    )
    monkeypatch.setattr(
        run_retrieval_eval,
        "_load_eval_settings",
        lambda: Settings(model_api_key="test-key"),
    )
    monkeypatch.setattr(run_retrieval_eval, "_build_ragas_llm", lambda _settings: "judge")

    rows = [
        {
            "id": "case-1",
            "question": "问题一",
            "contexts": ["上下文一"],
            "ground_truth": "答案一",
        },
        {
            "id": "case-2",
            "question": "问题二",
            "kb_ids": ["kb-a", "kb-b"],
            "contexts": ["上下文二"],
            "ground_truth": "答案二",
        },
    ]

    scored = run_retrieval_eval.evaluate_with_ragas(rows)

    assert captured["rows"] == [
        {
            "user_input": "问题一",
            "retrieved_contexts": ["上下文一"],
            "reference": "答案一",
        },
        {
            "user_input": "问题二",
            "retrieved_contexts": ["上下文二"],
            "reference": "答案二",
        },
    ]
    assert captured["dataset"] == "dataset"
    assert captured["kwargs"]["metrics"] == ["precision", "recall"]
    assert captured["kwargs"]["llm"] == "judge"
    assert scored[0]["context_recall"] == 0.4
    assert scored[0]["multi_kb"] is False
    assert scored[1]["multi_kb"] is True


def test_build_report_averages_metrics_and_lists_low_recall() -> None:
    report = build_report(
        {"name": "demo", "kb_id": "kb-a"},
        dataset_path=Path("eval/datasets/demo.json"),
        profile="balanced",
        scored_cases=[
            {
                "id": "case-1",
                "question": "问题一",
                "context_precision": 0.8,
                "context_recall": 0.4,
            },
            {
                "id": "case-2",
                "question": "问题二",
                "context_precision": 1.0,
                "context_recall": 0.9,
            },
        ],
        skipped=1,
    )

    assert report["summary"] == {
        "evaluated": 2,
        "skipped": 1,
        "context_precision": 0.9,
        "context_recall": 0.65,
    }
    assert [case["id"] for case in report["low_context_recall"]] == ["case-1"]
    assert [case["multi_kb"] for case in report["cases"]] == [False, False]


def test_build_dataset_keeps_case_kb_ids_when_multiple_kbs() -> None:
    dataset = build_dataset(
        "feedback",
        cases=[
            {"id": "one", "question": "一", "ground_truth": "答案", "kb_id": "kb-a"},
            {"id": "two", "question": "二", "ground_truth": "答案", "kb_id": "kb-b"},
        ],
        kb_id=None,
    )

    assert dataset["kb_id"] == ""
    assert dataset["cases"][0]["kb_ids"] == ["kb-a"]
    assert dataset["cases"][1]["kb_ids"] == ["kb-b"]
    assert "kb_id" not in dataset["cases"][0]
    assert "kb_id" not in dataset["cases"][1]
