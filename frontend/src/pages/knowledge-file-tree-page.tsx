import {
  ChevronDown,
  ChevronRight,
  Clipboard,
  Database,
  FileSearch,
  FileText,
  LoaderCircle,
  Pencil,
  RefreshCw,
  RotateCcw,
  Search,
  Trash2,
  Upload,
} from "lucide-react";
import { useEffect, useMemo, useState, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { EditKnowledgeBaseDialog } from "../components/knowledge-base/edit-knowledge-base-dialog";
import { StatusBadge } from "../components/knowledge-base/status-badge";
import { ConfirmDialog } from "../components/ui/confirm-dialog";
import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
import { HelpTooltip } from "../components/ui/help-tooltip";
import { Input } from "../components/ui/input";
import { getApiErrorMessage } from "../lib/api";
import { cn, formatDate } from "../lib/utils";
import { formatPlanQuota, tenantPlanMeta } from "../lib/tenant-plan";
import { authService } from "../services/authService";
import { documentService } from "../services/document";
import { knowledgeBaseService } from "../services/knowledge-base";
import type { AuthMeData, KnowledgeBaseTenantTree, KnowledgeBaseTreeDocument } from "../types";

type ConfirmTarget =
  | { kind: "document"; id: string; name: string }
  | { kind: "knowledge-base"; id: string; name: string };

function hasProcessingDocuments(tree: KnowledgeBaseTenantTree[] | undefined) {
  return Boolean(
    tree?.some((tenant) =>
      tenant.knowledge_bases.some((knowledgeBase) =>
        knowledgeBase.documents.some((document) => document.status === "PROCESSING"),
      ),
    ),
  );
}

function getKnowledgeBaseKey(kbId: string) {
  return kbId;
}

function getActionLinkClass() {
  return "inline-flex h-8 items-center gap-1.5 rounded-lg px-2.5 text-xs font-semibold text-muted transition-colors hover:bg-moss/8 hover:text-moss";
}

export function KnowledgeFileTreePage() {
  const queryClient = useQueryClient();
  const [keyword, setKeyword] = useState("");
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());
  const [notice, setNotice] = useState<string | null>(null);
  const [editingKnowledgeBaseId, setEditingKnowledgeBaseId] = useState<string | null>(null);
  const [confirmTarget, setConfirmTarget] = useState<ConfirmTarget | null>(null);

  const authQuery = useQuery<AuthMeData>({
    queryKey: ["auth-me"],
    queryFn: authService.fetchAuthMe,
  });
  const treeQuery = useQuery<KnowledgeBaseTenantTree[]>({
    queryKey: ["knowledge-base-tree", keyword.trim()],
    queryFn: () => knowledgeBaseService.fetchTree(keyword),
    refetchInterval: (query) => (hasProcessingDocuments(query.state.data) ? 3000 : false),
  });

  const deleteDocumentMutation = useMutation({
    mutationFn: (documentId: string) => documentService.deleteDocument(documentId),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["knowledge-base-tree"] });
      setNotice("文档已删除");
    },
    onError: (error) => setNotice(getApiErrorMessage(error)),
  });
  const reindexDocumentMutation = useMutation({
    mutationFn: (documentId: string) => documentService.reindexDocument(documentId),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["knowledge-base-tree"] });
      setNotice("已提交重试，文档正在后台索引");
    },
    onError: (error) => setNotice(getApiErrorMessage(error)),
  });
  const deleteKnowledgeBaseMutation = useMutation({
    mutationFn: (kbId: string) => knowledgeBaseService.deleteKnowledgeBase(kbId),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["knowledge-base-tree"] });
      setNotice("知识库及其文档已删除");
    },
    onError: (error) => setNotice(getApiErrorMessage(error)),
  });

  useEffect(() => {
    if (!notice) return;
    const timer = window.setTimeout(() => setNotice(null), 3500);
    return () => window.clearTimeout(timer);
  }, [notice]);

  const tree = treeQuery.data ?? [];
  const knowledgeBases = useMemo(
    () => tree.flatMap((tenant) => tenant.knowledge_bases),
    [tree],
  );
  const totals = useMemo(
    () => ({
      knowledgeBases: knowledgeBases.length,
      documents: knowledgeBases.reduce((count, knowledgeBase) => count + knowledgeBase.documents.length, 0),
    }),
    [knowledgeBases],
  );
  const processing = hasProcessingDocuments(tree);

  const toggle = (key: string) => {
    setCollapsed((current) => {
      const next = new Set(current);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };

  const copyKnowledgeBaseId = async (kbId: string) => {
    try {
      await navigator.clipboard.writeText(kbId);
      setNotice("知识库 ID 已复制");
    } catch {
      setNotice("复制失败，请手动选择 ID");
    }
  };

  const confirmDelete = () => {
    if (!confirmTarget) return;
    const target = confirmTarget;
    if (target.kind === "document") {
      deleteDocumentMutation.mutate(target.id, { onSettled: () => setConfirmTarget(null) });
    } else {
      deleteKnowledgeBaseMutation.mutate(target.id, { onSettled: () => setConfirmTarget(null) });
    }
  };

  return (
    <div className="space-y-7">
      <section className="flex flex-col justify-between gap-6 border-b border-line pb-7 lg:flex-row lg:items-end">
        <div>
          <div className="mb-3 flex items-center gap-2 text-xs font-bold uppercase tracking-[0.18em] text-moss">
            <span className="h-1.5 w-1.5 rounded-full bg-ember" />
            Knowledge inventory
          </div>
          <h1 className="text-balance text-3xl font-bold tracking-[-0.04em] text-ink md:text-[40px]">知识库列表</h1>
          <p className="mt-3 max-w-[680px] text-sm leading-6 text-muted md:text-[15px]">
            从当前租户读取知识库和文档状态，实时查看后台索引进度，完成编辑、重试与清理操作。
          </p>
        </div>
        <div className="flex flex-col items-stretch gap-3 lg:items-end">
          <div className="flex flex-wrap justify-end gap-2">
            {authQuery.isLoading && <Badge className="border-line bg-white text-muted">正在读取租户</Badge>}
            {authQuery.data && (
              <>
                <Badge className="border-moss/15 bg-moss/8 text-moss">租户：{authQuery.data.tenant_name}</Badge>
                <Badge className={tenantPlanMeta[authQuery.data.plan].className}>
                  <span className="h-1.5 w-1.5 rounded-full bg-current" />
                  套餐：{tenantPlanMeta[authQuery.data.plan].label}
                </Badge>
                <Badge className="border-line bg-white text-muted">
                  今日检索：{formatPlanQuota(authQuery.data.usage.retrieve_daily_count, authQuery.data.limits.retrieve_daily)}
                </Badge>
                <Badge className="border-line bg-white font-mono text-muted">ID：{authQuery.data.tenant_id}</Badge>
              </>
            )}
            {processing && (
              <Badge className="border-amber-200 bg-amber-50 text-amber-700">
                <LoaderCircle size={13} className="animate-spin" />
                有文档正在索引
              </Badge>
            )}
            {authQuery.isError && (
              <Badge className="border-red-200 bg-red-50 text-danger">{getApiErrorMessage(authQuery.error)}</Badge>
            )}
          </div>
          <div className="grid grid-cols-2 divide-x divide-line rounded-xl border border-line bg-white px-1 py-3 sm:min-w-[220px]">
            <SummaryStat label="知识库" value={totals.knowledgeBases} />
            <SummaryStat label="文档" value={totals.documents} />
          </div>
        </div>
      </section>

      <section className="rounded-2xl border border-line bg-white shadow-soft">
        <div className="flex flex-col justify-between gap-4 border-b border-line px-5 py-5 sm:flex-row sm:items-center sm:px-6">
          <div>
            <h2 className="text-base font-bold">全部知识库</h2>
            <p className="mt-1 text-xs text-muted">文档状态会自动刷新；全部进入终态后停止轮询。</p>
          </div>
          <div className="flex w-full items-center gap-2 sm:w-[320px]">
            <div className="relative min-w-0 flex-1">
              <Search size={16} className="pointer-events-none absolute left-3.5 top-1/2 -translate-y-1/2 text-muted" />
              <Input
                value={keyword}
                onChange={(event) => setKeyword(event.target.value)}
                className="pl-10"
                placeholder="按知识库名称搜索"
                aria-label="按知识库名称搜索"
              />
            </div>
            <Button
              type="button"
              variant="secondary"
              size="icon"
              title="刷新列表"
              aria-label="刷新列表"
              onClick={() => {
                void treeQuery.refetch();
                void authQuery.refetch();
              }}
              disabled={treeQuery.isFetching || authQuery.isFetching}
            >
              <RefreshCw size={16} className={treeQuery.isFetching || authQuery.isFetching ? "animate-spin" : ""} />
            </Button>
          </div>
        </div>

        {treeQuery.isLoading ? (
          <TreeState icon={<RefreshCw size={22} className="animate-spin" />} title="正在读取知识库" description="正在从后端加载当前租户的知识库和文档信息。" />
        ) : treeQuery.isError ? (
          <TreeState
            icon={<Database size={22} />}
            title="知识库列表加载失败"
            description={getApiErrorMessage(treeQuery.error)}
            action={
              <Button type="button" variant="secondary" size="sm" onClick={() => void treeQuery.refetch()}>
                <RefreshCw size={14} />
                重试
              </Button>
            }
          />
        ) : knowledgeBases.length === 0 ? (
          <TreeState
            icon={<Database size={22} />}
            title={keyword.trim() ? "没有匹配的知识库" : "还没有知识库"}
            description={keyword.trim() ? "换一个知识库名称关键词试试。" : "先从知识库上传页创建一个知识库。"}
            action={
              !keyword.trim() ? (
                <Button asChild size="sm">
                  <Link to="/">
                    <Upload size={14} />
                    去创建
                  </Link>
                </Button>
              ) : undefined
            }
          />
        ) : (
          <div role="table" aria-label="知识库与文档列表" className="divide-y divide-line">
            <div role="row" className="hidden grid-cols-[minmax(0,1.35fr)_minmax(160px,0.9fr)_minmax(100px,0.65fr)_minmax(280px,auto)] gap-4 bg-paper/70 px-5 py-3 text-[10px] font-bold uppercase tracking-[0.16em] text-muted sm:grid sm:px-6">
              <span role="columnheader">名称</span>
              <span role="columnheader">kb_id</span>
              <span role="columnheader">文档</span>
              <span role="columnheader" className="text-right">操作</span>
            </div>
            {knowledgeBases.map((knowledgeBase) => {
              const kbKey = getKnowledgeBaseKey(knowledgeBase.kb_id);
              const knowledgeBaseExpanded = !collapsed.has(kbKey);
              return (
                <div role="rowgroup" key={knowledgeBase.kb_id} className="bg-white">
                  <div role="row" className="grid gap-3 px-5 py-4 sm:grid-cols-[minmax(0,1.35fr)_minmax(160px,0.9fr)_minmax(100px,0.65fr)_minmax(280px,auto)] sm:items-center sm:gap-4 sm:px-6">
                    <div className="min-w-0">
                      <button
                        type="button"
                        className="flex min-w-0 items-start gap-2.5 text-left"
                        onClick={() => toggle(kbKey)}
                        aria-expanded={knowledgeBaseExpanded}
                      >
                        {knowledgeBaseExpanded ? <ChevronDown size={16} className="mt-0.5 shrink-0 text-moss" /> : <ChevronRight size={16} className="mt-0.5 shrink-0 text-muted" />}
                        <span className="min-w-0">
                          <span className="block truncate text-sm font-bold text-ink">{knowledgeBase.name}</span>
                          <span className="mt-1 block truncate text-xs text-muted">{knowledgeBase.description || "未填写描述"}</span>
                        </span>
                      </button>
                    </div>
                    <div className="pl-6 text-xs text-muted sm:pl-0">
                      <span className="mr-1.5 sm:hidden">kb_id:</span>
                      <code className="break-all font-mono text-[11px] text-ink/75">{knowledgeBase.kb_id}</code>
                    </div>
                    <div className="flex items-center gap-2 pl-6 text-xs text-muted sm:pl-0">
                      <FileText size={14} className="text-moss" />
                      {knowledgeBase.documents.length} 个文档
                    </div>
                    <div className="flex flex-wrap items-center justify-start gap-1 pl-6 sm:justify-end sm:pl-0">
                      <button
                        type="button"
                        title="复制 kb_id"
                        aria-label={`复制 ${knowledgeBase.name} 的 kb_id`}
                        className="inline-flex h-8 items-center gap-1.5 rounded-lg px-2.5 text-xs font-semibold text-muted transition-colors hover:bg-ink/5 hover:text-ink"
                        onClick={() => void copyKnowledgeBaseId(knowledgeBase.kb_id)}
                      >
                        <Clipboard size={14} />
                        <span className="hidden lg:inline">复制 ID</span>
                      </button>
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        className="h-8 px-2.5 text-xs"
                        title="编辑知识库"
                        aria-label={`编辑 ${knowledgeBase.name}`}
                        onClick={() => setEditingKnowledgeBaseId(knowledgeBase.kb_id)}
                      >
                        <Pencil size={14} />
                        <span className="hidden lg:inline">编辑</span>
                      </Button>
                      <Button
                        type="button"
                        variant="danger"
                        size="sm"
                        className="h-8 px-2.5 text-xs"
                        title="删除知识库"
                        aria-label={`删除 ${knowledgeBase.name}`}
                        onClick={() => setConfirmTarget({ kind: "knowledge-base", id: knowledgeBase.kb_id, name: knowledgeBase.name })}
                        disabled={deleteKnowledgeBaseMutation.isPending}
                      >
                        <Trash2 size={14} />
                        <span className="hidden lg:inline">删除</span>
                      </Button>
                      <Link
                        to={`/?kb_id=${encodeURIComponent(knowledgeBase.kb_id)}`}
                        className={getActionLinkClass()}
                        title="去上传"
                        aria-label={`去上传：${knowledgeBase.name}`}
                      >
                        <Upload size={14} />
                        <span className="hidden lg:inline">去上传</span>
                      </Link>
                      <Link
                        to={`/retrieve?kb_id=${encodeURIComponent(knowledgeBase.kb_id)}`}
                        className={cn(getActionLinkClass(), "bg-moss/5 text-moss hover:bg-moss/10")}
                        title="去检索"
                        aria-label={`去检索：${knowledgeBase.name}`}
                      >
                        <FileSearch size={14} />
                        <span className="hidden lg:inline">去检索</span>
                      </Link>
                    </div>
                  </div>

                  {knowledgeBaseExpanded && (
                    <div className="pb-4 pl-5 pr-5 sm:pl-14 sm:pr-6">
                      {knowledgeBase.documents.length === 0 ? (
                        <div className="rounded-xl border border-dashed border-line bg-paper/60 px-4 py-5 text-xs text-muted">暂无文档</div>
                      ) : (
                        <div className="overflow-hidden rounded-xl border border-line">
                          <div className="hidden grid-cols-[minmax(0,1.45fr)_minmax(115px,0.75fr)_minmax(150px,1fr)_70px_130px_minmax(150px,auto)] gap-3 bg-paper/70 px-4 py-2.5 text-[10px] font-bold uppercase tracking-[0.14em] text-muted xl:grid">
                            <span>文档</span>
                            <span>状态</span>
                            <span>document_id</span>
                            <span>chunk</span>
                            <span>创建时间</span>
                            <span className="text-right">操作</span>
                          </div>
                          <div className="divide-y divide-line">
                            {knowledgeBase.documents.map((document) => (
                              <DocumentRow
                                key={document.document_id}
                                document={document}
                                isReindexing={reindexDocumentMutation.isPending}
                                isDeleting={deleteDocumentMutation.isPending}
                                onReindex={() => reindexDocumentMutation.mutate(document.document_id)}
                                onDelete={() => setConfirmTarget({ kind: "document", id: document.document_id, name: document.title })}
                              />
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </section>

      <EditKnowledgeBaseDialog
        open={Boolean(editingKnowledgeBaseId)}
        kbId={editingKnowledgeBaseId}
        onClose={() => setEditingKnowledgeBaseId(null)}
        onSaved={() => setNotice("知识库已更新")}
      />

      {confirmTarget && (
        <ConfirmDialog
          open
          title={confirmTarget.kind === "knowledge-base" ? `删除知识库“${confirmTarget.name}”？` : `删除文档“${confirmTarget.name}”？`}
          description={confirmTarget.kind === "knowledge-base" ? "将删除该库及全部文档，不可恢复。" : "删除后无法恢复，需要重新上传原文件。"}
          isPending={confirmTarget.kind === "document" ? deleteDocumentMutation.isPending : deleteKnowledgeBaseMutation.isPending}
          onClose={() => setConfirmTarget(null)}
          onConfirm={confirmDelete}
        />
      )}

      {notice && (
        <div className="fixed bottom-5 right-5 z-50 max-w-[min(420px,calc(100vw-40px))] rounded-xl border border-line bg-white px-4 py-3 text-sm font-semibold text-ink shadow-popover" role="status">
          {notice}
        </div>
      )}
    </div>
  );
}

function DocumentRow({
  document,
  isReindexing,
  isDeleting,
  onReindex,
  onDelete,
}: {
  document: KnowledgeBaseTreeDocument;
  isReindexing: boolean;
  isDeleting: boolean;
  onReindex: () => void;
  onDelete: () => void;
}) {
  const errorMessage = document.status === "FAILED" ? document.error_message?.trim() : null;

  return (
    <div className="grid gap-3 px-4 py-3 sm:px-4 xl:grid-cols-[minmax(0,1.45fr)_minmax(115px,0.75fr)_minmax(150px,1fr)_70px_130px_minmax(150px,auto)] xl:items-center xl:gap-3">
      <div className="flex min-w-0 items-center gap-2">
        <FileText size={15} className="shrink-0 text-muted" />
        <div className="min-w-0">
          <span className="block truncate text-sm font-semibold text-ink" title={document.title}>{document.title}</span>
          <span className="mt-1 block text-xs text-muted xl:hidden">文档 · {formatDate(document.created_at)}</span>
        </div>
      </div>
      <div className="flex min-w-0 flex-wrap items-center gap-2 pl-6 xl:block xl:pl-0">
        <span className="text-[11px] font-semibold text-muted xl:hidden">状态</span>
        <StatusBadge status={document.status} />
        {errorMessage && (
          <div className="flex min-w-0 items-center gap-1 text-xs text-danger">
            <span className="max-w-[190px] truncate" title={errorMessage}>{errorMessage}</span>
            <HelpTooltip
              content={<span className="whitespace-pre-wrap break-words">{errorMessage}</span>}
              label="查看失败原因"
              placement="top"
            />
          </div>
        )}
      </div>
      <div className="min-w-0 pl-6 xl:pl-0">
        <span className="mr-1.5 text-[11px] text-muted xl:hidden">document_id:</span>
        <code className="break-all font-mono text-[11px] text-muted">{document.document_id}</code>
      </div>
      <div className="pl-6 text-xs font-semibold text-ink/75 xl:pl-0">
        <span className="mr-1.5 font-normal text-muted xl:hidden">chunk:</span>
        {document.chunk_count}
      </div>
      <div className="pl-6 text-xs text-muted xl:pl-0">{formatDate(document.created_at)}</div>
      <div className="flex flex-wrap items-center gap-2 pl-6 xl:justify-end xl:pl-0">
        {document.status === "FAILED" && (
          <Button
            type="button"
            variant="ghost"
            size="sm"
            className="h-8 px-2.5 text-xs text-moss hover:bg-moss/8"
            onClick={onReindex}
            disabled={isReindexing || isDeleting}
          >
            {isReindexing ? <LoaderCircle size={14} className="animate-spin" /> : <RotateCcw size={14} />}
            重试
          </Button>
        )}
        {document.status !== "PROCESSING" ? (
          <Button
            type="button"
            variant="danger"
            size="sm"
            className="h-8 px-2.5 text-xs"
            onClick={onDelete}
            disabled={isReindexing || isDeleting}
          >
            <Trash2 size={14} />
            删除
          </Button>
        ) : (
          <span className="text-xs text-muted">等待索引完成</span>
        )}
      </div>
    </div>
  );
}

function SummaryStat({ label, value }: { label: string; value: number }) {
  return (
    <div className="px-3 text-center">
      <p className="text-xl font-bold tracking-[-0.04em] text-ink">{value}</p>
      <p className="mt-1 text-[11px] font-semibold text-muted">{label}</p>
    </div>
  );
}

function TreeState({
  icon,
  title,
  description,
  action,
}: {
  icon: ReactNode;
  title: string;
  description: string;
  action?: ReactNode;
}) {
  return (
    <div className="grid min-h-[300px] place-items-center px-6 py-14 text-center">
      <div className="max-w-[420px]">
        <div className="mx-auto grid h-12 w-12 place-items-center rounded-2xl bg-moss/10 text-moss">{icon}</div>
        <h2 className="mt-4 text-base font-bold text-ink">{title}</h2>
        <p className="mt-2 text-sm leading-6 text-muted">{description}</p>
        {action && <div className="mt-5 flex justify-center">{action}</div>}
      </div>
    </div>
  );
}
