import {
  AlertCircle,
  CheckCircle2,
  ChevronDown,
  Clock3,
  FileSearch,
  Gauge,
  LoaderCircle,
  Search,
  Send,
  SlidersHorizontal,
  Sparkles,
  Star,
  TriangleAlert,
} from "lucide-react";
import { useEffect, useState, type FormEvent, type ReactNode } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import { useQuery } from "@tanstack/react-query";
import { useSearchParams } from "react-router-dom";
import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
import { Card } from "../components/ui/card";
import { HelpTooltip } from "../components/ui/help-tooltip";
import { Input } from "../components/ui/input";
import { Textarea } from "../components/ui/textarea";
import { getApiErrorMessage } from "../lib/api";
import { getProfileUpgradeMessage, tenantPlanMeta } from "../lib/tenant-plan";
import { authService } from "../services/authService";
import { ragService, type RetrievePayload } from "../services/rag";
import type {
  AuthMeData,
  QueryProcessingMetadata,
  RagRetrieveResponse,
  RetrieveProfile,
  RetrievalMode,
  RetrievedChunk,
} from "../types";

const retrievalModes: Array<{ value: RetrievalMode; label: string; description: string }> = [
  { value: "vector", label: "vector", description: "语义相似度" },
  { value: "bm25", label: "bm25", description: "关键词匹配" },
  { value: "hybrid", label: "hybrid", description: "RRF 融合" },
];

const retrievalProfiles: Array<{ value: RetrieveProfile; label: string; description: string }> = [
  { value: "speed", label: "追求速度", description: "vector 检索，响应最快" },
  { value: "balanced", label: "均衡", description: "hybrid 融合召回，适合日常查询" },
  { value: "quality", label: "追求质量", description: "hybrid + rerank + query 改写" },
  { value: "custom", label: "自定义", description: "手动调整高级检索参数" },
];

export function RetrievePage() {
  const [searchParams] = useSearchParams();
  const [kbId, setKbId] = useState(() => searchParams.get("kb_id") ?? "");
  const [query, setQuery] = useState("");
  const [profile, setProfile] = useState<RetrieveProfile>("balanced");
  const [topK, setTopK] = useState("5");
  const [mode, setMode] = useState<RetrievalMode>("hybrid");
  const [vectorTopK, setVectorTopK] = useState("20");
  const [bm25TopK, setBm25TopK] = useState("20");
  const [rrfK, setRrfK] = useState("60");
  const [rerankEnabled, setRerankEnabled] = useState(false);
  const [rerankTopN, setRerankTopN] = useState("5");
  const [queryRewriteEnabled, setQueryRewriteEnabled] = useState(false);
  const [result, setResult] = useState<RagRetrieveResponse | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [feedbackScore, setFeedbackScore] = useState<number | null>(null);
  const [feedbackComment, setFeedbackComment] = useState("");
  const [feedbackMessage, setFeedbackMessage] = useState<string | null>(null);
  const [feedbackError, setFeedbackError] = useState<string | null>(null);
  const [isSubmittingFeedback, setIsSubmittingFeedback] = useState(false);
  const [feedbackSubmitted, setFeedbackSubmitted] = useState(false);

  const authQuery = useQuery<AuthMeData>({
    queryKey: ["auth-me"],
    queryFn: authService.fetchAuthMe,
  });
  const tenantInfo = authQuery.data;
  const visibleRetrievalModes = tenantInfo?.features.hybrid_allowed
    ? retrievalModes
    : retrievalModes.filter((item) => item.value !== "hybrid");

  useEffect(() => {
    if (!tenantInfo) return;

    setProfile((current) =>
      tenantInfo.features.allowed_profiles.includes(current)
        ? current
        : tenantInfo.features.allowed_profiles[0] ?? "speed",
    );
    if (!tenantInfo.features.hybrid_allowed) {
      setMode((current) => (current === "hybrid" ? "vector" : current));
    }
    if (!tenantInfo.features.rerank_allowed) setRerankEnabled(false);
    if (!tenantInfo.features.query_rewrite_allowed) setQueryRewriteEnabled(false);
  }, [tenantInfo]);

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setError(null);

    const normalizedKbId = kbId.trim();
    const normalizedQuery = query.trim();
    if (!normalizedKbId || !normalizedQuery) {
      setError("kb_id 和 query 都是必填项。");
      return;
    }

    if (!tenantInfo) {
      setError(authQuery.isError ? getApiErrorMessage(authQuery.error) : "正在读取当前租户的套餐能力，请稍候。");
      return;
    }
    if (!tenantInfo.features.allowed_profiles.includes(profile)) {
      setError("当前套餐不支持所选检索档位，请先升级套餐。");
      return;
    }
    if (mode === "hybrid" && !tenantInfo.features.hybrid_allowed) {
      setError("当前套餐不支持 hybrid 检索，请调整自定义模式。");
      return;
    }

    const payload: RetrievePayload = {
      kb_id: normalizedKbId,
      user_id: "debug_user",
      query: normalizedQuery,
      profile,
    };

    if (profile === "custom") {
      const topKValue = parsePositiveInteger(topK);
      const vectorTopKValue = mode === "bm25" ? undefined : parsePositiveInteger(vectorTopK);
      const bm25TopKValue = mode === "vector" ? undefined : parsePositiveInteger(bm25TopK);
      const rrfKValue = mode === "hybrid" ? parsePositiveInteger(rrfK) : undefined;
      const rerankTopNValue = rerankEnabled ? parsePositiveInteger(rerankTopN) : undefined;
      if (
        !topKValue ||
        (mode !== "bm25" && !vectorTopKValue) ||
        (mode !== "vector" && !bm25TopKValue) ||
        (mode === "hybrid" && !rrfKValue) ||
        (rerankEnabled && !rerankTopNValue)
      ) {
        setError("检索参数必须是大于 0 的整数。");
        return;
      }

      const retrievalOptions: NonNullable<RetrievePayload["retrieval_options"]> = { mode };
      if (mode === "vector") retrievalOptions.vector_top_k = vectorTopKValue;
      if (mode === "bm25") retrievalOptions.bm25_top_k = bm25TopKValue;
      if (mode === "hybrid") {
        retrievalOptions.vector_top_k = vectorTopKValue;
        retrievalOptions.bm25_top_k = bm25TopKValue;
        retrievalOptions.rrf_k = rrfKValue;
      }

      payload.top_k = topKValue;
      payload.retrieval_options = retrievalOptions;
      if (rerankEnabled && tenantInfo.features.rerank_allowed) {
        payload.rerank_options = { enabled: true, top_n: rerankTopNValue ?? 5 };
      }
      if (queryRewriteEnabled && tenantInfo.features.query_rewrite_allowed) {
        payload.query_options = { enabled: true, strategy: "rewrite" };
      }
    }

    setResult(null);
    setFeedbackScore(null);
    setFeedbackComment("");
    setFeedbackMessage(null);
    setFeedbackError(null);
    setFeedbackSubmitted(false);
    setIsRunning(true);
    try {
      setResult(await ragService.retrieve(payload));
    } catch (requestError) {
      setError(getApiErrorMessage(requestError));
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div className="space-y-7">
      <section className="flex flex-col justify-between gap-6 border-b border-line pb-7 lg:flex-row lg:items-end">
        <div>
          <div className="mb-3 flex items-center gap-2 text-xs font-bold uppercase tracking-[0.18em] text-moss">
            <span className="h-1.5 w-1.5 rounded-full bg-ember" />
            Retrieval debugger
          </div>
          <h1 className="text-balance text-3xl font-bold tracking-[-0.04em] text-ink md:text-[40px]">检索调试</h1>
          <p className="mt-3 max-w-[680px] text-sm leading-6 text-muted md:text-[15px]">
            用一条真实 query 检查召回内容、各路分数、融合结果和服务端耗时。
          </p>
        </div>
        <div className="flex items-center gap-2 rounded-xl border border-moss/15 bg-moss/5 px-3.5 py-3 text-xs font-semibold text-moss">
          <FileSearch size={16} />
          user_id 固定为 debug_user
        </div>
      </section>

      <details open className="group rounded-2xl border border-line bg-white shadow-soft">
        <summary className="flex cursor-pointer list-none items-center justify-between gap-4 border-b border-line px-5 py-5 [&::-webkit-details-marker]:hidden sm:px-6">
          <span className="flex items-center gap-3">
            <span className="grid h-9 w-9 place-items-center rounded-xl bg-moss/10 text-moss">
              <SlidersHorizontal size={17} />
            </span>
            <span>
              <span className="block text-base font-bold">检索配置</span>
              <span className="mt-1 block text-xs text-muted">调整召回范围、融合参数与重排策略。</span>
            </span>
          </span>
          <ChevronDown size={18} className="text-muted transition-transform group-open:rotate-180" />
        </summary>
        <div className="space-y-6 px-5 py-6 sm:px-6">
          <div className="rounded-xl border border-line bg-paper/70 p-4 sm:p-5">
            <div className="flex flex-col justify-between gap-3 sm:flex-row sm:items-start">
              <div>
                <p className="text-[10px] font-bold uppercase tracking-[0.16em] text-muted">检索档位</p>
                <p className="mt-2 text-xs leading-5 text-muted">选择本次请求使用的服务端预设，套餐不支持的档位会锁定。</p>
              </div>
              {tenantInfo && (
                <Badge className={tenantPlanMeta[tenantInfo.plan].className}>当前套餐：{tenantPlanMeta[tenantInfo.plan].label}</Badge>
              )}
            </div>
            <div role="radiogroup" aria-label="检索档位" className="mt-4 grid gap-2 sm:grid-cols-2 xl:grid-cols-4">
              {retrievalProfiles.map((item) => {
                const active = profile === item.value;
                const allowed = Boolean(tenantInfo?.features.allowed_profiles.includes(item.value));
                const locked = Boolean(tenantInfo) && !allowed;
                const upgradeMessage = tenantInfo
                  ? getProfileUpgradeMessage(tenantInfo.plan, item.value, item.label)
                  : "";
                return (
                  <div key={item.value} className="relative min-w-0">
                    <button
                      type="button"
                      role="radio"
                      aria-checked={active}
                      disabled={!tenantInfo || !allowed}
                      onClick={() => {
                        setProfile(item.value);
                        setError(null);
                      }}
                      className={`flex min-h-[76px] w-full flex-col items-start justify-center rounded-xl border px-3.5 pr-10 text-left transition-colors disabled:cursor-not-allowed disabled:opacity-50 ${
                        active
                          ? "border-moss bg-moss/8 text-moss shadow-sm"
                          : "border-line bg-white text-muted hover:border-moss/35 hover:text-ink"
                      }`}
                    >
                      <span className="text-sm font-bold">{item.label}</span>
                      <span className="mt-1 text-[11px] font-medium leading-4 opacity-75">{item.description}</span>
                    </button>
                    {locked && (
                      <span className="absolute right-2 top-2">
                        <HelpTooltip content={upgradeMessage} label={`${item.label} 升级提示`} placement="top" />
                      </span>
                    )}
                  </div>
                );
              })}
            </div>
            {authQuery.isLoading && <p className="mt-3 text-xs text-muted">正在读取当前租户的套餐能力...</p>}
            {authQuery.isError && (
              <div className="mt-3 flex items-start gap-2 rounded-lg border border-red-200 bg-red-50 px-3 py-2.5 text-xs leading-5 text-danger" role="alert">
                <AlertCircle size={14} className="mt-0.5 shrink-0" />
                <span>{getApiErrorMessage(authQuery.error)}</span>
              </div>
            )}
          </div>

          <div className="rounded-xl border border-line bg-paper/70 p-4 sm:p-5">
            <p className="text-[10px] font-bold uppercase tracking-[0.16em] text-muted">检索范围</p>
            <div className="mt-4 grid gap-4">
              <FieldRow label="kb_id" required>
                <Input value={kbId} onChange={(event) => setKbId(event.target.value)} placeholder="知识库 ID" />
              </FieldRow>
            </div>
          </div>

          <div>
            <p className="border-b border-line pb-3 text-[10px] font-bold uppercase tracking-[0.16em] text-muted">检索参数</p>
            {profile === "custom" ? (
              <div className="mt-4 space-y-4">
                <FieldRow label="top_k" help="最终返回的 chunk 数量。数值越大，结果覆盖面越广，但响应内容也会更多。">
                  <Input type="number" min={1} value={topK} onChange={(event) => setTopK(event.target.value)} className="max-w-[180px]" />
                </FieldRow>
                <FieldRow label="检索模式" help="vector 适合语义相似问题，bm25 适合关键词匹配，hybrid 会融合两路结果。">
                  <div role="radiogroup" aria-label="检索模式" className="grid max-w-[620px] grid-cols-1 gap-2 sm:grid-cols-3">
                    {visibleRetrievalModes.map((item) => {
                      const active = mode === item.value;
                      return (
                        <button
                          key={item.value}
                          type="button"
                          role="radio"
                          aria-checked={active}
                          onClick={() => {
                            setMode(item.value);
                            setError(null);
                          }}
                          className={`flex min-h-[58px] flex-col items-start justify-center rounded-xl border px-3.5 text-left transition-colors ${
                            active ? "border-moss bg-moss/8 text-moss shadow-sm" : "border-line bg-white text-muted hover:border-moss/35 hover:text-ink"
                          }`}
                        >
                          <span className="text-sm font-bold">{item.label}</span>
                          <span className="mt-1 text-[11px] font-medium opacity-75">{item.description}</span>
                        </button>
                      );
                    })}
                  </div>
                </FieldRow>
                {mode === "hybrid" && (
                  <FieldRow label="hybrid 参数" help="hybrid 会分别召回向量和关键词结果，再用 RRF 合并排序。">
                    <div className="grid gap-3 sm:grid-cols-3">
                      <CompactNumberField label="vector_top_k" help="向量召回阶段保留的候选数量。" value={vectorTopK} onChange={setVectorTopK} />
                      <CompactNumberField label="bm25_top_k" help="BM25 关键词召回阶段保留的候选数量。" value={bm25TopK} onChange={setBm25TopK} />
                      <CompactNumberField label="rrf_k" help="RRF 融合中的平滑常数，用于降低单一路径高排名的影响。" value={rrfK} onChange={setRrfK} />
                    </div>
                  </FieldRow>
                )}
                <FieldRow label="rerank" help="对初步召回结果再次排序，通常能提升相关性，但会增加处理耗时。">
                  <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
                    <label className={`inline-flex h-11 w-fit items-center gap-2.5 rounded-xl border border-line bg-white px-3.5 text-sm font-semibold text-ink ${tenantInfo?.features.rerank_allowed ? "cursor-pointer" : "cursor-not-allowed opacity-50"}`}>
                      <input
                        type="checkbox"
                        checked={rerankEnabled}
                        disabled={!tenantInfo?.features.rerank_allowed}
                        onChange={(event) => setRerankEnabled(event.target.checked)}
                        className="h-4 w-4 accent-[#1e725c]"
                      />
                      启用重排
                    </label>
                    {tenantInfo && !tenantInfo.features.rerank_allowed && (
                      <HelpTooltip
                        content={getProfileUpgradeMessage(tenantInfo.plan, "quality", "重排")}
                        label="重排升级提示"
                      />
                    )}
                    {rerankEnabled && (
                      <label className="flex items-center gap-3 text-sm font-semibold text-ink">
                        <span className="whitespace-nowrap text-xs text-muted">rerank top_n</span>
                        <Input type="number" min={1} value={rerankTopN} onChange={(event) => setRerankTopN(event.target.value)} className="w-[120px]" />
                      </label>
                    )}
                  </div>
                </FieldRow>
                <FieldRow label="query 改写">
                  <div className="flex flex-wrap items-center gap-2.5">
                    <label className={`inline-flex h-11 w-fit items-center gap-2.5 rounded-xl border border-line bg-white px-3.5 text-sm font-semibold text-ink ${tenantInfo?.features.query_rewrite_allowed ? "cursor-pointer" : "cursor-not-allowed opacity-50"}`}>
                      <input
                        type="checkbox"
                        checked={queryRewriteEnabled}
                        disabled={!tenantInfo?.features.query_rewrite_allowed}
                        onChange={(event) => setQueryRewriteEnabled(event.target.checked)}
                        className="h-4 w-4 accent-[#1e725c]"
                      />
                      启用 query 改写
                    </label>
                    {tenantInfo && !tenantInfo.features.query_rewrite_allowed ? (
                      <HelpTooltip
                        content={getProfileUpgradeMessage(tenantInfo.plan, "quality", "query 改写")}
                        label="query 改写升级提示"
                      />
                    ) : (
                      <HelpTooltip
                        content="用 AI 把口语问题改成更好搜的说法；更慢、消耗 LLM，可对比开关效果。"
                        label="query 改写说明"
                      />
                    )}
                  </div>
                </FieldRow>
              </div>
            ) : (
              <div className="mt-4 flex flex-wrap items-center gap-2 rounded-xl border border-dashed border-line bg-paper/60 px-4 py-3 text-xs text-muted">
                <Badge className="border-moss/15 bg-moss/8 font-mono text-moss">profile: {profile}</Badge>
                <span>高级参数由服务端预设展开，本次请求只提交 profile。</span>
              </div>
            )}
          </div>
        </div>
      </details>

      <Card>
        <div className="border-b border-line px-5 py-5 sm:px-6">
          <div className="flex items-start gap-3">
            <div className="grid h-9 w-9 shrink-0 place-items-center rounded-xl bg-[#fff3df] text-ember">
              <Search size={17} />
            </div>
            <div>
              <h2 className="text-base font-bold">检索与召回</h2>
              <p className="mt-1 text-xs text-muted">输入问题后查看每个 chunk 的召回来源和评分明细。</p>
            </div>
          </div>
        </div>
        <form onSubmit={handleSubmit} className="px-5 py-6 sm:px-6">
          <FieldRow label="query" required alignTop>
            <Textarea value={query} onChange={(event) => setQuery(event.target.value)} placeholder="例如：LangGraph 如何保存执行状态？" className="min-h-[132px]" />
          </FieldRow>
          {error && (
            <div className="mt-5 flex items-start gap-2.5 rounded-xl border border-red-200 bg-red-50 px-3.5 py-3 text-xs leading-5 text-danger" role="alert">
              <AlertCircle size={15} className="mt-0.5 shrink-0" />
              <span>{error}</span>
            </div>
          )}
          <div className="mt-5 flex justify-end border-t border-line pt-5">
            <Button type="submit" disabled={isRunning || authQuery.isLoading || authQuery.isError || !tenantInfo}>
              {isRunning ? <LoaderCircle size={16} className="animate-spin" /> : <Search size={16} />}
              {isRunning ? "检索中..." : authQuery.isLoading ? "读取套餐..." : "检索"}
            </Button>
          </div>
        </form>

        {result?.metadata.trace_id ? (
          <FeedbackPanel
            score={feedbackScore}
            comment={feedbackComment}
            message={feedbackMessage}
            error={feedbackError}
            isSubmitting={isSubmittingFeedback}
            submitted={feedbackSubmitted}
            onScoreChange={(value) => {
              setFeedbackScore(value);
              setFeedbackMessage(null);
              setFeedbackError(null);
            }}
            onCommentChange={(value) => {
              setFeedbackComment(value);
              setFeedbackMessage(null);
              setFeedbackError(null);
            }}
            onSubmit={async () => {
              if (!feedbackScore) {
                setFeedbackError("请先选择 1～5 分。");
                return;
              }
              const traceId = result.metadata.trace_id;
              if (!traceId) return;
              setIsSubmittingFeedback(true);
              setFeedbackMessage(null);
              setFeedbackError(null);
              try {
                const payload = {
                  trace_id: traceId,
                  score: feedbackScore,
                  ...(result.metadata.log_id ? { log_id: result.metadata.log_id } : {}),
                  ...(feedbackComment.trim() ? { comment: feedbackComment.trim() } : {}),
                };
                await ragService.submitFeedback(payload);
                setFeedbackMessage(`已提交，已评 ${feedbackScore} 分`);
                setFeedbackSubmitted(true);
              } catch (requestError) {
                setFeedbackError(getApiErrorMessage(requestError));
              } finally {
                setIsSubmittingFeedback(false);
              }
            }}
          />
        ) : result ? (
          <div className="border-t border-line px-5 py-4 text-xs text-muted sm:px-6">
            Langfuse 未启用，反馈暂不可用。
          </div>
        ) : null}

        <div className="border-t border-line px-5 py-5 sm:px-6">
          <div className="flex items-center justify-between gap-3">
            <div>
              <h3 className="text-sm font-bold">召回结果</h3>
              <p className="mt-1 text-xs text-muted">
                {isRunning ? "正在检索，请稍候" : result ? `返回 ${result.retrieved_chunks.length} 个 chunk` : "执行一次检索后显示结果"}
              </p>
            </div>
            {result && <Badge className="border-moss/15 bg-moss/8 text-moss">{result.metadata.retrieval?.mode ?? "retrieval"}</Badge>}
          </div>
          {isRunning ? (
            <RetrievalLoadingState />
          ) : result ? (
            <div className="mt-4 space-y-3">
              {result.retrieved_chunks.length === 0 ? (
                <div className="rounded-xl border border-dashed border-line bg-paper/60 px-5 py-10 text-center text-sm text-muted">没有召回内容。</div>
              ) : (
                result.retrieved_chunks.map((chunk, index) => <RetrievedChunkRow key={chunk.chunk_id} chunk={chunk} index={index} />)
              )}
            </div>
          ) : (
            <div className="mt-4 grid min-h-[160px] place-items-center rounded-xl border border-dashed border-line bg-paper/60 px-6 py-8 text-center">
              <div>
                <FileSearch size={24} className="mx-auto text-muted/50" />
                <p className="mt-3 text-sm font-semibold text-muted">等待 query</p>
                <p className="mt-1 text-xs text-muted/75">召回的内容、分数和来源会在这里展开。</p>
              </div>
            </div>
          )}
        </div>
      </Card>

      <RunSummary result={result} isRunning={isRunning} />
    </div>
  );
}

function RetrievalLoadingState() {
  return (
    <div
      className="mt-4 overflow-hidden rounded-xl border border-moss/20 bg-moss/5 px-5 py-8"
      role="status"
      aria-live="polite"
    >
      <div className="flex items-center gap-3">
        <span className="grid h-10 w-10 shrink-0 place-items-center rounded-xl bg-white text-moss shadow-sm">
          <LoaderCircle size={20} className="animate-spin" />
        </span>
        <div>
          <p className="text-sm font-bold text-ink">正在检索</p>
          <p className="mt-1 text-xs text-muted">正在召回候选片段并整理排序结果</p>
        </div>
      </div>
      <div className="mt-5 space-y-2" aria-hidden="true">
        <div className="h-2 w-11/12 animate-pulse rounded-full bg-moss/15" />
        <div className="h-2 w-4/5 animate-pulse rounded-full bg-moss/10" />
        <div className="h-2 w-2/3 animate-pulse rounded-full bg-moss/10" />
      </div>
    </div>
  );
}

function FeedbackPanel({
  score,
  comment,
  message,
  error,
  isSubmitting,
  submitted,
  onScoreChange,
  onCommentChange,
  onSubmit,
}: {
  score: number | null;
  comment: string;
  message: string | null;
  error: string | null;
  isSubmitting: boolean;
  submitted: boolean;
  onScoreChange: (score: number) => void;
  onCommentChange: (comment: string) => void;
  onSubmit: () => Promise<void>;
}) {
  return (
    <div className="border-t border-line bg-paper/45 px-5 py-5 sm:px-6">
      <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-start">
        <div>
          <p className="text-[10px] font-bold uppercase tracking-[0.16em] text-muted">检索反馈</p>
          <p className="mt-1 text-xs text-muted">评分会挂到本次检索的 Langfuse trace。</p>
        </div>
        <div role="radiogroup" aria-label="检索反馈评分" className="flex items-center gap-1">
          {[1, 2, 3, 4, 5].map((value) => {
            const active = score !== null && value <= score;
            return (
              <button
                key={value}
                type="button"
                role="radio"
                aria-checked={score === value}
                aria-label={`${value} 分`}
                title={`${value} 分`}
                disabled={isSubmitting || submitted}
                onClick={() => onScoreChange(value)}
                className={`grid h-9 w-9 place-items-center rounded-lg transition-colors disabled:cursor-not-allowed disabled:opacity-50 ${
                  active
                    ? "text-amber-400 hover:bg-amber-50"
                    : "text-muted/45 hover:bg-ink/5 hover:text-muted"
                }`}
              >
                <Star size={23} fill={active ? "currentColor" : "transparent"} strokeWidth={1.8} />
              </button>
            );
          })}
        </div>
      </div>
      <div className="mt-4 grid gap-3 sm:grid-cols-[minmax(0,1fr)_auto] sm:items-end">
        <Textarea
          value={comment}
          onChange={(event) => onCommentChange(event.target.value)}
          placeholder="备注（可选），例如：排第三的 chunk 才是对的"
          className="min-h-[88px] bg-white"
          maxLength={2000}
          disabled={isSubmitting || submitted}
        />
        <Button type="button" size="sm" disabled={isSubmitting || submitted} onClick={onSubmit}>
          {isSubmitting ? <LoaderCircle size={15} className="animate-spin" /> : submitted ? <CheckCircle2 size={15} /> : <Send size={15} />}
          {isSubmitting ? "提交中..." : submitted ? "已提交" : "提交反馈"}
        </Button>
      </div>
      {message && <p className="mt-3 text-xs font-semibold text-moss" role="status">{message}</p>}
      {error && <p className="mt-3 text-xs font-semibold text-danger" role="alert">{error}</p>}
    </div>
  );
}

function FieldRow({
  label,
  required,
  help,
  alignTop = false,
  children,
}: {
  label: string;
  required?: boolean;
  help?: ReactNode;
  alignTop?: boolean;
  children: ReactNode;
}) {
  return (
    <div className={`grid gap-2 sm:grid-cols-[100px_minmax(0,1fr)] sm:gap-4 ${alignTop ? "sm:items-start" : "sm:items-center"}`}>
      <div className={`text-sm font-bold text-ink sm:pt-2 sm:text-right ${alignTop ? "sm:pt-3" : ""}`}>
        <span className="inline-flex items-center gap-1">
          {label}
          {required && <span className="text-ember">*</span>}
          {help && <HelpTooltip content={help} label={`${label} 参数说明`} placement="right" />}
        </span>
      </div>
      <div className="min-w-0">{children}</div>
    </div>
  );
}

function CompactNumberField({
  label,
  help,
  value,
  onChange,
}: {
  label: string;
  help?: ReactNode;
  value: string;
  onChange: (value: string) => void;
}) {
  return (
    <label className="block">
      <span className="mb-1.5 flex items-center gap-1 text-[11px] font-semibold text-muted">
        {label}
        {help && <HelpTooltip content={help} label={`${label} 参数说明`} />}
      </span>
      <Input type="number" min={1} value={value} onChange={(event) => onChange(event.target.value)} />
    </label>
  );
}

const retrievalSourceLabels: Record<RetrievedChunk["retrieval_source"], string> = {
  vector: "语义召回",
  bm25: "关键词召回",
  hybrid: "混合召回",
};

function RetrievedChunkRow({ chunk, index }: { chunk: RetrievedChunk; index: number }) {
  const chunkType = getChunkTypeLabel(chunk.metadata?.chunk_type);

  return (
    <article className="overflow-hidden rounded-xl border border-line bg-white">
      <div className="grid lg:grid-cols-[minmax(240px,0.38fr)_minmax(0,1.62fr)]">
        <aside className="min-w-0 border-b border-line bg-paper/65 p-5 sm:p-6 lg:border-b-0 lg:border-r">
          <div className="flex items-center justify-between gap-3">
            <Badge className="border-ember/20 bg-ember/8 text-ember">#{index + 1}</Badge>
            <Badge className="border-line bg-white text-muted">{retrievalSourceLabels[chunk.retrieval_source]}</Badge>
          </div>
          <h4 className="mt-4 break-words text-[15px] font-bold leading-6 text-ink">{chunk.title}</h4>

          <div className="mt-5 divide-y divide-line border-y border-line bg-white/65 px-3.5">
            <ScoreMetric label="综合排序分" help="该 chunk 在最终结果中的综合排序分数。" score={chunk.score} rank={index + 1} />
            <ScoreMetric label="语义相似度" help="向量检索返回的相似度分数及原始排名。" score={chunk.vector_score} rank={chunk.vector_rank} />
            <ScoreMetric label="关键词匹配分" help="BM25 关键词检索返回的匹配分数及原始排名。" score={chunk.bm25_score} rank={chunk.bm25_rank} />
            <ScoreMetric label="重排分" help="重排模型对该 chunk 的最终评分及排名。" score={chunk.rerank_score} rank={chunk.rerank_score == null ? null : index + 1} />
          </div>

          <div className="mt-5 space-y-3 border-t border-line pt-4">
            <ChunkMetadataRow label="文档 ID" value={chunk.document_id} mono />
            <ChunkMetadataRow label="切片 ID" value={chunk.chunk_id} mono />
            {chunk.metadata?.heading_path && <ChunkMetadataRow label="所属章节" value={chunk.metadata.heading_path} />}
          </div>
        </aside>

        <section className="min-w-0 p-5 sm:p-6">
          <div className="flex items-start justify-between gap-4">
            <div>
              <p className="text-[10px] font-bold uppercase tracking-[0.16em] text-muted">召回内容</p>
              <p className="mt-1 text-xs text-muted">按 Markdown / GFM 格式渲染</p>
            </div>
            {chunkType && <Badge className="border-moss/15 bg-moss/8 text-moss">{chunkType}</Badge>}
          </div>
          <div className="mt-5 overflow-hidden rounded-xl bg-paper/65 px-4 py-4 sm:px-5 sm:py-5">
            <MarkdownContent content={chunk.content} chunkType={chunk.metadata?.chunk_type} />
          </div>
        </section>
      </div>
    </article>
  );
}

function ChunkMetadataRow({ label, value, mono = false }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="min-w-0">
      <span className="block text-[11px] text-muted">{label}</span>
      <span
        className={`mt-0.5 block break-words leading-5 text-ink/85 ${mono ? "font-mono text-[11px]" : "text-xs font-semibold"}`}
        title={value}
      >
        {value}
      </span>
    </div>
  );
}

function ScoreMetric({
  label,
  help,
  score,
  rank,
}: {
  label: string;
  help?: ReactNode;
  score?: number | null;
  rank?: number | null;
}) {
  return (
    <div className="flex items-center justify-between gap-3 py-2.5">
      <MetricLabel label={label} help={help} />
      <span className="shrink-0 text-right">
        <span className="block font-mono text-sm font-bold text-ink">{score == null ? "—" : formatScore(score)}</span>
        {rank != null && <span className="mt-0.5 block text-[10px] text-muted">第 {rank} 名</span>}
      </span>
    </div>
  );
}

function MetricLabel({ label, help }: { label: string; help?: ReactNode }) {
  return (
    <span className="flex items-center gap-1 text-xs font-semibold text-muted">
      {label}
      {help && <HelpTooltip content={help} label={`${label} 指标说明`} />}
    </span>
  );
}

function MarkdownContent({ content, chunkType }: { content: string; chunkType?: unknown }) {
  const markdown = normalizeChunkContent(content, chunkType);

  if (!markdown) {
    return <p className="text-sm text-muted">该切片没有可展示的内容。</p>;
  }

  return (
    <div className="min-w-0 break-words text-[13px] leading-7 text-ink/85">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
        {markdown}
      </ReactMarkdown>
    </div>
  );
}

const markdownComponents: Components = {
  h1: ({ children }) => <h3 className="mb-4 text-lg font-bold leading-7 text-ink">{children}</h3>,
  h2: ({ children }) => <h3 className="mb-3 text-base font-bold leading-6 text-ink">{children}</h3>,
  h3: ({ children }) => <h4 className="mb-2 text-sm font-bold leading-6 text-ink">{children}</h4>,
  h4: ({ children }) => <h5 className="mb-2 text-sm font-semibold leading-6 text-ink">{children}</h5>,
  p: ({ children }) => <p className="mb-3 last:mb-0">{children}</p>,
  ul: ({ children }) => <ul className="mb-4 list-disc space-y-1 pl-5 last:mb-0">{children}</ul>,
  ol: ({ children }) => <ol className="mb-4 list-decimal space-y-1 pl-5 last:mb-0">{children}</ol>,
  li: ({ children }) => <li className="pl-1">{children}</li>,
  blockquote: ({ children }) => <blockquote className="mb-4 border-l-2 border-moss/40 pl-4 text-muted">{children}</blockquote>,
  hr: () => <hr className="my-5 border-line" />,
  strong: ({ children }) => <strong className="font-bold text-ink">{children}</strong>,
  em: ({ children }) => <em className="text-ink/75">{children}</em>,
  a: ({ href, children }) => (
    <a className="font-semibold text-moss underline decoration-moss/30 underline-offset-2 hover:text-moss-dark" href={href} target="_blank" rel="noreferrer">
      {children}
    </a>
  ),
  pre: ({ children }) => <pre className="mb-4 overflow-x-auto rounded-lg border border-line bg-white px-4 py-3 text-xs leading-6">{children}</pre>,
  code: ({ className, children }) => (
    <code className={className ? `font-mono text-[12px] ${className}` : "rounded bg-white px-1.5 py-0.5 font-mono text-[12px] text-moss-dark"}>
      {children}
    </code>
  ),
  table: ({ children }) => (
    <div className="mb-4 overflow-x-auto rounded-lg border border-line bg-white last:mb-0">
      <table className="min-w-full border-collapse text-left text-[12px] leading-5">{children}</table>
    </div>
  ),
  thead: ({ children }) => <thead className="bg-[#e9eef0] text-ink">{children}</thead>,
  tbody: ({ children }) => <tbody className="divide-y divide-line">{children}</tbody>,
  tr: ({ children }) => <tr className="align-top">{children}</tr>,
  th: ({ children }) => <th className="border-b border-line px-3 py-2.5 font-bold">{children}</th>,
  td: ({ children }) => <td className="px-3 py-2.5 text-ink/80">{children}</td>,
};

function getChunkTypeLabel(value: unknown) {
  if (typeof value !== "string" || !value.trim()) return null;
  return (
    {
      section: "正文",
      table: "表格",
      table_part: "表格片段",
    }[value] ?? value
  );
}

function normalizeChunkContent(content: string, chunkType: unknown) {
  const trimmed = content.trim();
  if (!trimmed || (chunkType !== "table" && chunkType !== "table_part")) return trimmed;

  const lines = trimmed.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
  if (lines.length < 2) return trimmed;

  const headerIndex = lines.findIndex((line) => line.startsWith("【表头】"));
  if (headerIndex >= 0) {
    const headerCells = splitTableCells(lines[headerIndex].replace("【表头】", ""));
    if (headerCells.length > 0) {
      return buildMarkdownTable(headerCells, lines.slice(headerIndex + 1), lines.slice(0, headerIndex));
    }
  }

  if (lines[0].includes("|") && !isMarkdownTableSeparator(lines[1])) {
    const headerCells = splitTableCells(lines[0]);
    if (headerCells.length > 1) return buildMarkdownTable(headerCells, lines.slice(1));
  }

  return trimmed;
}

function buildMarkdownTable(headerCells: string[], rows: string[], prefix: string[] = []) {
  const normalizedRows = rows.filter(Boolean).map((row) => (row.startsWith("|") ? row : `| ${row} |`));
  return [
    ...prefix,
    `| ${headerCells.join(" | ")} |`,
    `| ${headerCells.map(() => "---").join(" | ")} |`,
    ...normalizedRows,
  ].join("\n");
}

function splitTableCells(line: string) {
  const normalized = line.trim().replace(/^\|/, "").replace(/\|$/, "");
  return normalized.split("|").map((cell) => cell.trim());
}

function isMarkdownTableSeparator(line: string) {
  const cells = splitTableCells(line);
  return cells.length > 0 && cells.every((cell) => /^:?-{3,}:?$/.test(cell));
}

function RunSummary({ result, isRunning }: { result: RagRetrieveResponse | null; isRunning: boolean }) {
  const queryProcessing = result?.metadata.query_processing;

  return (
    <section className="rounded-2xl border border-line bg-white shadow-soft">
      <div className="flex items-start gap-3 border-b border-line px-5 py-5 sm:px-6">
        <div className="grid h-9 w-9 shrink-0 place-items-center rounded-xl bg-moss/10 text-moss">
          <Gauge size={17} />
        </div>
        <div>
          <h2 className="text-base font-bold">运行摘要</h2>
          <p className="mt-1 text-xs text-muted">只读展示本次服务端返回的 metadata。</p>
        </div>
      </div>
      {isRunning ? (
        <div className="px-5 py-8 sm:px-6" role="status" aria-live="polite">
          <div className="flex items-center gap-2 text-sm font-semibold text-ink">
            <LoaderCircle size={16} className="animate-spin text-moss" />
            正在更新运行摘要
          </div>
          <p className="mt-2 text-xs text-muted">本次检索完成后显示召回数量、耗时和降级状态。</p>
        </div>
      ) : !result ? (
        <div className="px-5 py-8 text-sm text-muted sm:px-6">完成一次检索后，这里会显示服务端耗时、召回数量和降级状态。</div>
      ) : (
        <div className="space-y-4 px-5 py-5 sm:px-6">
          <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
            <SummaryOverview label="检索模式" code="mode" value={result.metadata.retrieval?.mode ?? "—"} />
            <SummaryOverview label="结果上限" code="top_k" value={result.metadata.top_k ?? "—"} suffix="个" />
            <SummaryOverview label="最终返回" code="count" value={result.retrieved_chunks.length} suffix="个" />
            <SummaryOverview
              label="服务端耗时"
              code="latency"
              value={result.metadata.latency_ms ?? "—"}
              suffix="ms"
              icon={<Clock3 size={14} className="text-moss" />}
            />
          </div>
          <div className="grid overflow-hidden rounded-xl border border-line bg-paper/60 sm:grid-cols-3 sm:divide-x sm:divide-line">
            <SummaryMetric
              label="向量召回"
              code="vector"
              description="语义相似候选"
              value={result.metadata.retrieval?.vector_count ?? "—"}
            />
            <SummaryMetric
              label="关键词召回"
              code="bm25"
              description="关键词匹配候选"
              value={result.metadata.retrieval?.bm25_count ?? "—"}
            />
            <SummaryMetric
              label="融合结果"
              code="fused"
              description="合并后的候选"
              value={result.metadata.retrieval?.fused_count ?? "—"}
            />
          </div>
          {queryProcessing && <QueryProcessingSummary processing={queryProcessing} />}
          <div className="flex flex-wrap items-center gap-2 text-xs text-muted">
            {result.metadata.rerank?.enabled ? (
              <Badge className="border-moss/15 bg-moss/8 text-moss">
                rerank: {result.metadata.rerank.model || result.metadata.rerank.provider || "enabled"}
              </Badge>
            ) : (
              <Badge className="border-line bg-paper text-muted">rerank: disabled</Badge>
            )}
            {result.metadata.retrieval?.degraded || result.metadata.rerank?.degraded || queryProcessing?.degraded ? (
              <Badge className="border-amber-200 bg-amber-50 text-amber-700">
                <TriangleAlert size={13} />
                degraded
              </Badge>
            ) : (
              <Badge className="border-emerald-200 bg-emerald-50 text-emerald-700">
                <CheckCircle2 size={13} />
                healthy
              </Badge>
            )}
            {result.metadata.retrieval?.degraded_reason && <span>{result.metadata.retrieval.degraded_reason}</span>}
            {result.metadata.rerank?.error && <span>{result.metadata.rerank.error}</span>}
          </div>
        </div>
      )}
    </section>
  );
}

function QueryProcessingSummary({ processing }: { processing: QueryProcessingMetadata }) {
  const hasExpandedSearch = processing.search_query !== processing.effective_query;

  return (
    <div className="rounded-xl border border-line bg-white p-4 text-xs">
      <div className="flex items-center gap-2 font-bold text-ink">
        <Sparkles size={14} className="text-moss" />
        query processing
      </div>
      <div className="mt-3 space-y-2.5">
        <QueryTextRow label="原话" value={processing.raw_query} />
        <QueryTextRow label="实际检索句" value={processing.effective_query} />
        <QueryTextRow
          label="最终检索句"
          value={processing.search_query}
          emphasized={hasExpandedSearch}
        />
      </div>
      <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-2 text-muted">
        <span>改写耗时 {processing.rewrite_latency_ms}ms</span>
        {processing.synonym_applied && (
          <span>词表命中：{processing.synonym_expansions.join("、") || "已命中"}</span>
        )}
      </div>
      {processing.degraded && (
        <div className="mt-3 flex flex-wrap items-center gap-2 text-amber-700">
          <TriangleAlert size={13} />
          <span>改写失败，已用原话检索</span>
          {processing.degraded_reason && <span>({processing.degraded_reason})</span>}
        </div>
      )}
    </div>
  );
}

function QueryTextRow({
  label,
  value,
  emphasized = false,
}: {
  label: string;
  value: string;
  emphasized?: boolean;
}) {
  return (
    <div className={`grid gap-1.5 sm:grid-cols-[88px_minmax(0,1fr)] sm:items-start sm:gap-3 ${emphasized ? "rounded-lg border border-ember/25 bg-ember/5 px-2.5 py-2" : ""}`}>
      <span className="font-semibold text-muted">{label}</span>
      <span className="break-words leading-5 text-ink">{value}</span>
    </div>
  );
}

function SummaryOverview({
  label,
  code,
  value,
  suffix,
  icon,
}: {
  label: string;
  code: string;
  value: number | string;
  suffix?: string;
  icon?: ReactNode;
}) {
  return (
    <div className="rounded-xl border border-line bg-paper/60 px-3.5 py-3">
      <div className="flex items-center gap-2 text-[10px] font-bold uppercase tracking-[0.12em] text-muted">
        {icon}
        <span>{code}</span>
      </div>
      <div className="mt-1.5 flex items-baseline justify-between gap-3">
        <span className="text-xs font-semibold text-ink">{label}</span>
        <span className="font-mono text-sm font-bold text-ink">
          {value}
          {suffix && <span className="ml-1 text-[11px] font-semibold text-muted">{suffix}</span>}
        </span>
      </div>
    </div>
  );
}

function SummaryMetric({
  label,
  code,
  description,
  value,
}: {
  label: string;
  code: string;
  description: string;
  value: number | string;
}) {
  return (
    <div className="flex items-center gap-3 px-4 py-3.5">
      <div className="min-w-0">
        <div className="flex items-baseline gap-2">
          <span className="text-sm font-bold text-ink">{label}</span>
          <span className="font-mono text-[10px] font-semibold uppercase tracking-[0.1em] text-muted">{code}</span>
        </div>
        <p className="mt-1 text-xs text-muted">{description}</p>
      </div>
      <span className="shrink-0 font-mono text-xl font-bold text-ink">{value}</span>
    </div>
  );
}

function parsePositiveInteger(value: string) {
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : undefined;
}

function formatScore(value: number) {
  return value.toFixed(3);
}
