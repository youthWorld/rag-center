import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { RetrievePage } from "./retrieve-page";
import type { AuthMeData, RagRetrieveResponse, ResearchData } from "../types";

const mocks = vi.hoisted(() => ({
  auth: vi.fn(),
  tree: vi.fn(),
  versions: vi.fn(),
  retrieve: vi.fn(),
  research: vi.fn(),
  feedback: vi.fn(),
}));

vi.mock("../services/authService", () => ({ authService: { fetchAuthMe: mocks.auth } }));
vi.mock("../services/knowledge-base", () => ({
  knowledgeBaseService: { fetchTree: mocks.tree, fetchIndexVersions: mocks.versions },
}));
vi.mock("../services/rag", () => ({
  ragService: { retrieve: mocks.retrieve, research: mocks.research, submitFeedback: mocks.feedback },
}));

const pro: AuthMeData = {
  tenant_id: "tenant-1",
  tenant_name: "Test",
  plan: "pro",
  features: {
    allowed_profiles: ["speed", "balanced", "quality", "custom"],
    hybrid_allowed: true,
    rerank_allowed: true,
    query_rewrite_allowed: true,
    evidence_allowed: true,
    research_allowed: true,
  },
  limits: {
    retrieve_qps: 10,
    retrieve_daily: 1000,
    max_kb: 20,
    max_kb_per_retrieve: 5,
    max_documents_per_kb: 100,
    max_processing_documents: 10,
  },
  usage: { kb_count: 1, retrieve_daily_count: 0 },
};

const single: RagRetrieveResponse = {
  query: "first question",
  kb_id: "kb-1",
  kb_ids: ["kb-1"],
  retrieved_chunks: [],
  metadata: { trace_id: "single-trace", log_id: "single-log", latency_ms: 4 },
};

const research: ResearchData = {
  query: "first question",
  kb_id: "kb-1",
  kb_ids: ["kb-1"],
  answer: "结论来自证据 [E1]。",
  plan: {
    degraded: false,
    aspects: [{ aspect_id: "a1", description: "问题要点" }],
    initial_queries: [{ query_id: "q1", query: "检索语句", aspect_ids: ["a1"] }],
  },
  rounds: [
    { round: 1, purpose: "initial", candidate_count: 1, new_chunk_count: 1, latency_ms: 12,
      tasks: [{ query_id: "q1", query: "检索语句", aspect_ids: ["a1"], round: 1, success: true,
        latency_ms: 12, chunk_count: 1, new_chunk_count: 1, degraded: false }] },
    { round: 2, purpose: "supplement", candidate_count: 1, new_chunk_count: 0, latency_ms: 10,
      tasks: [{ query_id: "q2", query: "补证语句", aspect_ids: ["a1"], round: 2, success: true,
        latency_ms: 10, chunk_count: 1, new_chunk_count: 0, degraded: false }] },
  ],
  evidence_pack: {
    status: "complete",
    missing_aspects: [],
    groups: [{ aspect: "问题要点", covered: true, evidence: [{ evidence_id: "E1", role: "core" }] }],
    items: [{ evidence_id: "E1", chunk_id: "chunk-1", kb_id: "kb-1", document_id: "doc-1",
      index_version: "v1", title: "原文标题", content: "真实证据正文", source: "vector",
      rounds: [1, 2], query_ids: ["q1", "q2"] }],
  },
  metadata: {
    research_id: "research-1", log_id: "research-log", trace_id: "research-trace",
    profile: "research_fixed", tenant_plan: "pro", index_versions: { "kb-1": "v1" },
    round_count: 2, retrieval_task_count: 2, llm_call_count: 3, input_tokens: 120,
    output_tokens: 50, latency_ms: 100, stop_reason: "max_rounds", degraded: false,
    first_round_missing_aspect_ids: ["a1"],
    errors: [], stages: [{ stage: "finalizer", role: "strong", model: "qwen3.8-flash",
      latency_ms: 12, input_tokens: 20, output_tokens: 10, degraded: false }],
    planner: {}, decider: {}, finalizer: { degraded: false },
  },
};

function mount() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={client}><MemoryRouter initialEntries={["/retrieve"]} future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
    <RetrievePage />
  </MemoryRouter></QueryClientProvider>);
}

async function ask(user: ReturnType<typeof userEvent.setup>) {
  await user.type(screen.getByPlaceholderText(/LangGraph/), "first question");
  await user.click(screen.getByRole("button", { name: /^检索$/ }));
}

describe("RetrievePage research mode", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mocks.auth.mockResolvedValue(pro);
    mocks.tree.mockResolvedValue([{ tenant_id: "tenant-1", knowledge_bases: [
      { kb_id: "kb-1", name: "测试库", created_at: "2026-01-01", documents: [] },
    ] }]);
    mocks.versions.mockResolvedValue({ active_index_version: "v1", versions: [] });
    mocks.retrieve.mockResolvedValue(single);
    mocks.research.mockResolvedValue(research);
    mocks.feedback.mockResolvedValue({ feedback_id: "score-1" });
  });

  it("uses the ordinary endpoint by default and the restricted research payload in Research mode", async () => {
    const user = userEvent.setup();
    mount();
    await screen.findByLabelText("选择 测试库");
    await ask(user);
    expect(mocks.retrieve).toHaveBeenCalledWith(expect.objectContaining({ kb_id: "kb-1", query: "first question", profile: "balanced" }));
    expect(mocks.research).not.toHaveBeenCalled();
    await user.click(screen.getByRole("tab", { name: /Research/ }));
    expect(screen.getByTestId("research-fixed-config")).toBeInTheDocument();
    expect(screen.queryByRole("radiogroup", { name: "检索档位" })).not.toBeInTheDocument();
    expect(screen.queryByLabelText("索引版本")).not.toBeInTheDocument();
    expect(screen.queryByText("证据编排")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /^检索$/ }));
    expect(mocks.research).toHaveBeenCalledWith({ kb_id: "kb-1", query: "first question", user_id: "debug_user" });
    expect(screen.queryByText("召回结果")).not.toBeInTheDocument();
  });

  it("blocks Research for a non-Pro plan without invoking the endpoint", async () => {
    mocks.auth.mockResolvedValue({ ...pro, plan: "standard", features: { ...pro.features, research_allowed: false } });
    const user = userEvent.setup();
    mount();
    await screen.findByLabelText("选择 测试库");
    await user.click(screen.getByRole("tab", { name: /Research/ }));
    await ask(user);
    expect(await screen.findByText(/当前套餐不支持 Research/)).toBeInTheDocument();
    expect(mocks.research).not.toHaveBeenCalled();
  });

  it("shows the answer, plan, rounds, usage and a working evidence anchor", async () => {
    mocks.research.mockResolvedValue({ ...research, answer: "**支持 Markdown**。[E1]" });
    const user = userEvent.setup();
    mount();
    await screen.findByLabelText("选择 测试库");
    await user.click(screen.getByRole("tab", { name: /Research/ }));
    await ask(user);
    const result = await screen.findByTestId("research-result");
    expect(within(result).getAllByText(/检索语句/)).toHaveLength(2);
    expect(within(result).getByText(/补证语句/)).toBeInTheDocument();
    expect(within(result).getByText(/真实证据正文/)).toBeInTheDocument();
    expect(within(result).getByText(/3 次 Research LLM 调用/)).toBeInTheDocument();
    expect(within(result).getByRole("link", { name: "[E1]" })).toHaveAttribute("href", "#research-evidence-E1");
    expect(within(result).getByText("支持 Markdown").tagName).toBe("STRONG");
    expect(Array.from(result.querySelectorAll("h2")).slice(0, 3).map((node) => node.textContent))
      .toEqual(["Query Plan", "检索轮次与补证", "Research 回答"]);
    expect(document.getElementById("research-evidence-E1")).toHaveTextContent("真实证据正文");
  });

  it("shows which first-round gaps each supplement query targets even when the final pack is complete", async () => {
    const aspects = [
      { aspect_id: "A1", description: "现金退款" },
      { aspect_id: "A2", description: "积分退回" },
      { aspect_id: "A3", description: "积分有效期" },
    ];
    mocks.research.mockResolvedValue({
      ...research,
      plan: { ...research.plan, aspects, initial_queries: [
        { query_id: "Q1", query: "跨库退款", aspect_ids: ["A1", "A2", "A3"] },
      ] },
      rounds: [
        { ...research.rounds[0], tasks: [
          { ...research.rounds[0].tasks[0], query_id: "Q1", aspect_ids: ["A1", "A2", "A3"] },
        ] },
        { ...research.rounds[1], tasks: [
          { ...research.rounds[1].tasks[0], query_id: "Q4", aspect_ids: ["A3"] },
          { ...research.rounds[1].tasks[0], query_id: "Q5", query: "积分补证", aspect_ids: ["A2", "A3"] },
        ] },
      ],
      evidence_pack: { ...research.evidence_pack, groups: aspects.map((aspect) => ({
        aspect: aspect.description, covered: true, evidence: [{ evidence_id: "E1", role: "core" as const }],
      })) },
      metadata: { ...research.metadata, first_round_missing_aspect_ids: ["A1", "A2", "A3"] },
    } satisfies ResearchData);
    const user = userEvent.setup();
    mount();
    await screen.findByLabelText("选择 测试库");
    await user.click(screen.getByRole("tab", { name: /Research/ }));
    await ask(user);
    const reason = await screen.findByTestId("research-supplement-reason-2");
    expect(screen.getByTestId("research-first-round-missing"))
      .toHaveTextContent("首轮缺失要点：A1：现金退款、A2：积分退回、A3：积分有效期");
    expect(reason).toHaveTextContent("补证原因：首轮缺失要点：A1：现金退款、A2：积分退回、A3：积分有效期");
    expect(reason).toHaveTextContent("本轮查询绑定的要点：A2：积分退回、A3：积分有效期");
    const q4 = screen.getByText("Q4：补证语句").closest("li");
    const q5 = screen.getByText("Q5：积分补证").closest("li");
    expect(q4).not.toBeNull();
    expect(q5).not.toBeNull();
    expect(within(q4!).getByText("绑定要点：A3：积分有效期")).toBeInTheDocument();
    expect(within(q5!).getByText("绑定要点：A2：积分退回、A3：积分有效期")).toBeInTheDocument();
    expect(screen.getByText("证据状态：complete")).toBeInTheDocument();
  });

  it("does not invent a supplement reason when there is no second round", async () => {
    mocks.research.mockResolvedValue({
      ...research, rounds: [research.rounds[0]],
      metadata: { ...research.metadata, round_count: 1, stop_reason: "evidence_complete",
        first_round_missing_aspect_ids: [] },
    } satisfies ResearchData);
    const user = userEvent.setup();
    mount();
    await screen.findByLabelText("选择 测试库");
    await user.click(screen.getByRole("tab", { name: /Research/ }));
    await ask(user);
    await screen.findByTestId("research-result");
    expect(screen.getByTestId("research-first-round-missing")).toHaveTextContent("首轮缺失要点：无");
    expect(screen.queryByText(/补证原因：/)).not.toBeInTheDocument();
    expect(screen.queryByTestId("research-supplement-reason-2")).not.toBeInTheDocument();
  });

  it("labels decider failures as unverified rather than confirmed missing evidence", async () => {
    mocks.research.mockResolvedValue({
      ...research, rounds: [research.rounds[0]],
      metadata: { ...research.metadata, round_count: 1, degraded: true,
        first_round_missing_aspect_ids: ["a1"], decider: { degraded: true } },
    } satisfies ResearchData);
    const user = userEvent.setup();
    mount();
    await screen.findByLabelText("选择 测试库");
    await user.click(screen.getByRole("tab", { name: /Research/ }));
    await ask(user);
    expect(await screen.findByTestId("research-first-round-missing"))
      .toHaveTextContent("首轮未确认覆盖的要点（Decider 降级）：a1：问题要点");
    expect(screen.queryByText(/补证原因：/)).not.toBeInTheDocument();
  });

  it("shows finalizer fallback with null answer and validated evidence", async () => {
    mocks.research.mockResolvedValue({ ...research, answer: null,
      metadata: { ...research.metadata, degraded: true, stop_reason: "degraded",
        finalizer: { degraded: true, error: "LLM_TIMEOUT" } } });
    const user = userEvent.setup();
    mount();
    await screen.findByLabelText("选择 测试库");
    await user.click(screen.getByRole("tab", { name: /Research/ }));
    await ask(user);
    expect(await screen.findByText(/Finalizer 已降级/)).toBeInTheDocument();
    expect(screen.getByText(/本次未生成回答/)).toBeInTheDocument();
    expect(document.getElementById("research-evidence-E1")).toHaveTextContent("真实证据正文");
  });

  it("clears stale results on query change and never restores them after a page refresh", async () => {
    const user = userEvent.setup();
    const page = mount();
    await screen.findByLabelText("选择 测试库");
    await user.click(screen.getByRole("tab", { name: /Research/ }));
    await ask(user);
    await screen.findByTestId("research-result");
    fireEvent.change(screen.getByPlaceholderText(/LangGraph/), { target: { value: "new question" } });
    expect(screen.queryByTestId("research-result")).not.toBeInTheDocument();
    page.unmount();
    mount();
    expect(screen.queryByTestId("research-result")).not.toBeInTheDocument();
  });

  it("allows repeated feedback updates with the same trace and log IDs in both modes", async () => {
    const user = userEvent.setup();
    mount();
    await screen.findByLabelText("选择 测试库");
    await ask(user);
    await screen.findByRole("radiogroup", { name: "检索反馈评分" });
    await user.click(screen.getByRole("radio", { name: "3 分" }));
    await user.click(screen.getByRole("button", { name: "提交反馈" }));
    expect(mocks.feedback).toHaveBeenCalledWith({ trace_id: "single-trace", log_id: "single-log", score: 3 });
    await screen.findByText(/反馈已保存/);
    await user.click(screen.getByRole("radio", { name: "5 分" }));
    await user.click(screen.getByRole("button", { name: "提交反馈" }));
    expect(mocks.feedback).toHaveBeenLastCalledWith({ trace_id: "single-trace", log_id: "single-log", score: 5 });

    await user.click(screen.getByRole("tab", { name: /Research/ }));
    await user.click(screen.getByRole("button", { name: /^检索$/ }));
    await screen.findByTestId("research-result");
    await user.click(screen.getByRole("radio", { name: "4 分" }));
    await user.click(screen.getByRole("button", { name: "提交反馈" }));
    expect(mocks.feedback).toHaveBeenLastCalledWith({ trace_id: "research-trace", log_id: "research-log", score: 4 });
  });
});
