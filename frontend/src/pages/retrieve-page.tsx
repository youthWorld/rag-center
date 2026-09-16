import {
  AlertCircle,
  CheckCircle2,
  ChevronDown,
  Clock3,
  FileSearch,
  Gauge,
  LoaderCircle,
  Search,
  SlidersHorizontal,
  Sparkles,
  TriangleAlert,
} from "lucide-react";
import { useState, type FormEvent, type ReactNode } from "react";
import { useSearchParams } from "react-router-dom";
import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
import { HelpTooltip } from "../components/ui/help-tooltip";
import { Input } from "../components/ui/input";
import { Textarea } from "../components/ui/textarea";
import { getApiErrorMessage } from "../lib/api";
import { ragService, type RetrievePayload } from "../services/rag";
import type {
  QueryProcessingMetadata,
  RagRetrieveResponse,
  RetrievalMode,
  RetrievedChunk,
} from "../types";

const retrievalModes: Array<{ value: RetrievalMode; label: string; description: string }> = [
  { value: "vector", label: "vector", description: "语义相似度" },
  { value: "bm25", label: "bm25", description: "关键词匹配" },
  { value: "hybrid", label: "hybrid", description: "RRF 融合" },
];

export function RetrievePage() {
  const [searchParams] = useSearchParams();
  const [kbId, setKbId] = useState(() => searchParams.get("kb_id") ?? "");
  const [query, setQuery] = useState("");
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

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setError(null);

    const normalizedKbId = kbId.trim();
    const normalizedQuery = query.trim();
    if (!normalizedKbId || !normalizedQuery) {
      setError("kb_id 和 query 都是必填项。");
      return;
    }

    const topKValue = parsePositiveInteger(topK);
    const vectorTopKValue = parsePositiveInteger(vectorTopK);
    const bm25TopKValue = parsePositiveInteger(bm25TopK);
    const rrfKValue = parsePositiveInteger(rrfK);
    const rerankTopNValue = parsePositiveInteger(rerankTopN);
    if (!topKValue || !vectorTopKValue || !bm25TopKValue || !rrfKValue || (rerankEnabled && !rerankTopNValue)) {
      setError("检索参数必须是大于 0 的整数。");
      return;
    }

    const retrievalOptions: RetrievePayload["retrieval_options"] = { mode };
    if (mode === "vector") retrievalOptions.vector_top_k = vectorTopKValue;
    if (mode === "bm25") retrievalOptions.bm25_top_k = bm25TopKValue;
    if (mode === "hybrid") {
      retrievalOptions.vector_top_k = vectorTopKValue;
      retrievalOptions.bm25_top_k = bm25TopKValue;
      retrievalOptions.rrf_k = rrfKValue;
    }

    const payload: RetrievePayload = {
      kb_id: normalizedKbId,
      user_id: "debug_user",
      query: normalizedQuery,
      top_k: topKValue,
      retrieval_options: retrievalOptions,
    };
    if (rerankEnabled) {
      payload.rerank_options = { enabled: true, top_n: rerankTopNValue ?? 5 };
    }
    if (queryRewriteEnabled) {
      payload.query_options = { enabled: true, strategy: "rewrite" };
    }

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
            <p className="text-[10px] font-bold uppercase tracking-[0.16em] text-muted">检索范围</p>
            <div className="mt-4 grid gap-4">
              <FieldRow label="kb_id" required>
                <Input value={kbId} onChange={(event) => setKbId(event.target.value)} placeholder="知识库 ID" />
              </FieldRow>
            </div>
          </div>

          <div>
            <p className="border-b border-line pb-3 text-[10px] font-bold uppercase tracking-[0.16em] text-muted">检索参数</p>
            <div className="mt-4 space-y-4">
              <FieldRow label="top_k" help="最终返回的 chunk 数量。数值越大，结果覆盖面越广，但响应内容也会更多。">
                <Input type="number" min={1} value={topK} onChange={(event) => setTopK(event.target.value)} className="max-w-[180px]" />
              </FieldRow>
              <FieldRow label="检索模式" help="vector 适合语义相似问题，bm25 适合关键词匹配，hybrid 会融合两路结果。">
                <div role="radiogroup" aria-label="检索模式" className="grid max-w-[620px] grid-cols-1 gap-2 sm:grid-cols-3">
                  {retrievalModes.map((item) => {
                    const active = mode === item.value;
                    return (
                      <button
                        key={item.value}
                        type="button"
                        role="radio"
                        aria-checked={active}
                        onClick={() => {
                          setMode(item.value);
                          setResult(null);
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
                  <label className="inline-flex h-11 w-fit cursor-pointer items-center gap-2.5 rounded-xl border border-line bg-white px-3.5 text-sm font-semibold text-ink">
                    <input
                      type="checkbox"
                      checked={rerankEnabled}
                      onChange={(event) => setRerankEnabled(event.target.checked)}
                      className="h-4 w-4 accent-[#1e725c]"
                    />
                    启用重排
                  </label>
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
                  <label className="inline-flex h-11 w-fit cursor-pointer items-center gap-2.5 rounded-xl border border-line bg-white px-3.5 text-sm font-semibold text-ink">
                    <input
                      type="checkbox"
                      checked={queryRewriteEnabled}
                      onChange={(event) => setQueryRewriteEnabled(event.target.checked)}
                      className="h-4 w-4 accent-[#1e725c]"
                    />
                    启用 query 改写
                  </label>
                  <HelpTooltip
                    content="用 AI 把口语问题改成更好搜的说法；更慢、消耗 LLM，可对比开关效果。"
                    label="query 改写说明"
                  />
                </div>
              </FieldRow>
            </div>
          </div>
        </div>
      </details>

      <section className="rounded-2xl border border-line bg-white shadow-soft">
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
            <Button type="submit" disabled={isRunning}>
              {isRunning ? <LoaderCircle size={16} className="animate-spin" /> : <Search size={16} />}
              {isRunning ? "检索中..." : "检索"}
            </Button>
          </div>
        </form>

        <div className="border-t border-line px-5 py-5 sm:px-6">
          <div className="flex items-center justify-between gap-3">
            <div>
              <h3 className="text-sm font-bold">召回结果</h3>
              <p className="mt-1 text-xs text-muted">{result ? `返回 ${result.retrieved_chunks.length} 个 chunk` : "执行一次检索后显示结果"}</p>
            </div>
            {result && <Badge className="border-moss/15 bg-moss/8 text-moss">{result.metadata.retrieval?.mode ?? "retrieval"}</Badge>}
          </div>
          {result ? (
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
      </section>

      <RunSummary result={result} />
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

function RetrievedChunkRow({ chunk, index }: { chunk: RetrievedChunk; index: number }) {
  return (
    <article className="overflow-hidden rounded-xl border border-line bg-white">
      <div className="flex flex-col gap-4 px-4 py-4 sm:px-5">
        <div className="flex flex-col justify-between gap-3 lg:flex-row lg:items-start">
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <Badge className="border-ember/20 bg-ember/8 text-ember">#{index + 1}</Badge>
              <span className="font-mono text-xs font-bold text-ink">score={formatScore(chunk.score)}</span>
              <Badge className="border-line bg-paper text-muted">{chunk.retrieval_source}</Badge>
            </div>
            <h4 className="mt-3 text-sm font-bold text-ink">{chunk.title}</h4>
            <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted">
              <span>document_id: <code className="font-mono text-ink/75">{chunk.document_id}</code></span>
              <span>chunk_id: <code className="font-mono text-ink/75">{chunk.chunk_id}</code></span>
            </div>
          </div>
          <div className="grid grid-cols-2 gap-x-5 gap-y-2 text-xs lg:min-w-[360px] lg:grid-cols-4">
             <ScoreMetric label="vector" help="向量相似度召回分数及其原始排名。" score={chunk.vector_score} rank={chunk.vector_rank} />
             <ScoreMetric label="bm25" help="BM25 关键词召回分数及其原始排名。" score={chunk.bm25_score} rank={chunk.bm25_rank} />
             <ScoreMetric label="rerank" help="重排模型对该 chunk 的最终评分及排名。" score={chunk.rerank_score} rank={chunk.rerank_score == null ? null : index + 1} />
             <div>
               <MetricLabel label="position" help="该 chunk 在最终返回结果中的位置。" />
               <span className="mt-1 block font-mono font-bold text-ink">#{index + 1}</span>
             </div>
          </div>
        </div>
        <details className="group rounded-lg border border-line bg-paper/60">
          <summary className="flex cursor-pointer list-none items-center justify-between gap-3 px-3.5 py-2.5 text-xs font-semibold text-muted hover:text-ink [&::-webkit-details-marker]:hidden">
            查看 content
            <ChevronDown size={15} className="transition-transform group-open:rotate-180" />
          </summary>
          <div className="border-t border-line px-3.5 py-3 text-xs leading-6 text-ink/80">
            <p className="whitespace-pre-wrap break-words">{chunk.content}</p>
          </div>
        </details>
      </div>
    </article>
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
    <div>
      <MetricLabel label={label} help={help} />
      <span className="mt-1 block font-mono font-bold text-ink">
        {score == null ? "—" : `${formatScore(score)} · #${rank ?? "—"}`}
      </span>
    </div>
  );
}

function MetricLabel({ label, help }: { label: string; help?: ReactNode }) {
  return (
    <span className="flex items-center gap-1 text-[10px] font-bold uppercase tracking-[0.12em] text-muted">
      {label}
      {help && <HelpTooltip content={help} label={`${label} 指标说明`} />}
    </span>
  );
}

function RunSummary({ result }: { result: RagRetrieveResponse | null }) {
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
      {!result ? (
        <div className="px-5 py-8 text-sm text-muted sm:px-6">完成一次检索后，这里会显示服务端耗时、召回数量和降级状态。</div>
      ) : (
        <div className="space-y-4 px-5 py-5 sm:px-6">
          <div className="flex flex-wrap items-center gap-x-3 gap-y-2 text-sm font-semibold text-ink">
            <span>mode={result.metadata.retrieval?.mode ?? "—"}</span>
            <span className="text-muted/50">|</span>
            <span>top_k={result.metadata.top_k ?? "—"}</span>
            <span className="text-muted/50">|</span>
            <span>count={result.retrieved_chunks.length}</span>
            <span className="text-muted/50">|</span>
            <span className="inline-flex items-center gap-1.5"><Clock3 size={14} className="text-moss" />服务端耗时 {result.metadata.latency_ms ?? "—"}ms</span>
          </div>
          <div className="grid gap-3 rounded-xl border border-line bg-paper/60 p-4 text-xs sm:grid-cols-3">
            <SummaryMetric label="vector" value={result.metadata.retrieval?.vector_count ?? "—"} />
            <SummaryMetric label="bm25" value={result.metadata.retrieval?.bm25_count ?? "—"} />
            <SummaryMetric label="fused" value={result.metadata.retrieval?.fused_count ?? "—"} />
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

function SummaryMetric({ label, value }: { label: string; value: number | string }) {
  return (
    <div className="flex items-center justify-between gap-3">
      <span className="font-mono text-muted">{label}</span>
      <span className="font-bold text-ink">{value}</span>
    </div>
  );
}

function parsePositiveInteger(value: string) {
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : null;
}

function formatScore(value: number) {
  return value.toFixed(3);
}
