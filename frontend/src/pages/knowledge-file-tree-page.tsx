import {
  ChevronDown,
  ChevronRight,
  Clipboard,
  Database,
  FileSearch,
  FileText,
  RefreshCw,
  Search,
  Upload,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
import { Input } from "../components/ui/input";
import { getApiErrorMessage } from "../lib/api";
import { cn, formatDate } from "../lib/utils";
import { knowledgeBaseService } from "../services/knowledge-base";
import type { KnowledgeBaseTenantTree } from "../types";

function getKnowledgeBaseKey(tenantId: string, kbId: string) {
  return `${tenantId}:${kbId}`;
}

function getActionLinkClass() {
  return "inline-flex h-8 items-center gap-1.5 rounded-lg px-2.5 text-xs font-semibold text-muted transition-colors hover:bg-moss/8 hover:text-moss";
}

export function KnowledgeFileTreePage() {
  const [keyword, setKeyword] = useState("");
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());
  const [notice, setNotice] = useState<string | null>(null);
  const treeQuery = useQuery<KnowledgeBaseTenantTree[]>({
    queryKey: ["knowledge-base-tree", keyword.trim()],
    queryFn: () => knowledgeBaseService.fetchTree(keyword),
  });

  useEffect(() => {
    if (!notice) return;
    const timer = window.setTimeout(() => setNotice(null), 3000);
    return () => window.clearTimeout(timer);
  }, [notice]);

  const tree = treeQuery.data ?? [];
  const totals = useMemo(
    () =>
      tree.reduce(
        (summary, tenant) => {
          summary.tenants += 1;
          summary.knowledgeBases += tenant.knowledge_bases.length;
          summary.documents += tenant.knowledge_bases.reduce(
            (count, knowledgeBase) => count + knowledgeBase.documents.length,
            0,
          );
          return summary;
        },
        { tenants: 0, knowledgeBases: 0, documents: 0 },
      ),
    [tree],
  );

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
            从后端读取所有租户、知识库和已完成索引的文档，快速回到上传工作区或验证检索效果。
          </p>
        </div>
        <div className="grid grid-cols-3 divide-x divide-line rounded-xl border border-line bg-white px-1 py-3 sm:min-w-[330px]">
          <SummaryStat label="租户" value={totals.tenants} />
          <SummaryStat label="知识库" value={totals.knowledgeBases} />
          <SummaryStat label="成功文档" value={totals.documents} />
        </div>
      </section>

      <section className="rounded-2xl border border-line bg-white shadow-soft">
        <div className="flex flex-col justify-between gap-4 border-b border-line px-5 py-5 sm:flex-row sm:items-center sm:px-6">
          <div>
            <h2 className="text-base font-bold">全部知识库</h2>
            <p className="mt-1 text-xs text-muted">文档仅显示状态为 SUCCESS 的记录。</p>
          </div>
          <div className="flex w-full items-center gap-2 sm:w-[320px]">
            <div className="relative min-w-0 flex-1">
              <Search size={16} className="pointer-events-none absolute left-3.5 top-1/2 -translate-y-1/2 text-muted" />
              <Input
                value={keyword}
                onChange={(event) => setKeyword(event.target.value)}
                className="pl-10"
                placeholder="按 tenant_id 搜索"
                aria-label="按租户标识搜索"
              />
            </div>
            <Button
              type="button"
              variant="secondary"
              size="icon"
              title="刷新列表"
              aria-label="刷新列表"
              onClick={() => void treeQuery.refetch()}
              disabled={treeQuery.isFetching}
            >
              <RefreshCw size={16} className={treeQuery.isFetching ? "animate-spin" : ""} />
            </Button>
          </div>
        </div>

        {treeQuery.isLoading ? (
          <TreeState icon={<RefreshCw size={22} className="animate-spin" />} title="正在读取知识库" description="正在从后端加载租户和文档信息。" />
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
        ) : tree.length === 0 ? (
          <TreeState
            icon={<Database size={22} />}
            title={keyword.trim() ? "没有匹配的租户" : "还没有知识库"}
            description={keyword.trim() ? "换一个 tenant_id 关键词试试。" : "先从知识库上传页创建一个知识库。"}
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
          <div role="table" aria-label="知识库树" className="divide-y divide-line">
            <div role="row" className="hidden grid-cols-[minmax(0,1.45fr)_minmax(180px,1fr)_minmax(110px,0.65fr)_minmax(220px,auto)] gap-4 bg-paper/70 px-5 py-3 text-[10px] font-bold uppercase tracking-[0.16em] text-muted sm:grid sm:px-6">
              <span role="columnheader">名称</span>
              <span role="columnheader">标识</span>
              <span role="columnheader">文档</span>
              <span role="columnheader" className="text-right">操作</span>
            </div>
            {tree.map((tenant) => {
              const tenantKey = `tenant:${tenant.tenant_id}`;
              const tenantExpanded = !collapsed.has(tenantKey);
              const documentCount = tenant.knowledge_bases.reduce(
                (count, knowledgeBase) => count + knowledgeBase.documents.length,
                0,
              );
              return (
                <div role="rowgroup" key={tenant.tenant_id}>
                  <div role="row" className="bg-[#f7faf8] px-5 py-4 sm:px-6">
                    <button
                      type="button"
                      className="flex w-full items-center justify-between gap-4 text-left"
                      onClick={() => toggle(tenantKey)}
                      aria-expanded={tenantExpanded}
                    >
                      <span className="flex min-w-0 items-center gap-2.5">
                        {tenantExpanded ? <ChevronDown size={17} className="shrink-0 text-moss" /> : <ChevronRight size={17} className="shrink-0 text-muted" />}
                        <span className="truncate text-sm font-bold text-ink">{tenant.tenant_id}</span>
                        <Badge className="border-moss/15 bg-moss/8 text-moss">
                          {tenant.knowledge_bases.length} 个知识库
                        </Badge>
                      </span>
                      <span className="hidden shrink-0 text-xs text-muted sm:inline">{documentCount} 个成功文档</span>
                    </button>
                  </div>

                  {tenantExpanded && (
                    <div className="divide-y divide-line/80">
                      {tenant.knowledge_bases.map((knowledgeBase) => {
                        const kbKey = getKnowledgeBaseKey(tenant.tenant_id, knowledgeBase.kb_id);
                        const knowledgeBaseExpanded = !collapsed.has(kbKey);
                        return (
                          <div role="rowgroup" key={knowledgeBase.kb_id} className="bg-white">
                            <div role="row" className="grid gap-3 px-5 py-4 sm:grid-cols-[minmax(0,1.45fr)_minmax(180px,1fr)_minmax(110px,0.65fr)_minmax(220px,auto)] sm:items-center sm:gap-4 sm:px-6">
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
                                {knowledgeBase.documents.length} 个成功文档
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
                                <Link
                                  to={`/?tenant_id=${encodeURIComponent(tenant.tenant_id)}&kb_id=${encodeURIComponent(knowledgeBase.kb_id)}`}
                                  className={getActionLinkClass()}
                                >
                                  <Upload size={14} />
                                  去上传
                                </Link>
                                <Link
                                  to={`/retrieve?tenant_id=${encodeURIComponent(tenant.tenant_id)}&kb_id=${encodeURIComponent(knowledgeBase.kb_id)}`}
                                  className={cn(getActionLinkClass(), "bg-moss/5 text-moss hover:bg-moss/10")}
                                >
                                  <FileSearch size={14} />
                                  去检索
                                </Link>
                              </div>
                            </div>

                            {knowledgeBaseExpanded && (
                              <div className="pb-4 pl-5 pr-5 sm:pl-14 sm:pr-6">
                                {knowledgeBase.documents.length === 0 ? (
                                  <div className="rounded-xl border border-dashed border-line bg-paper/60 px-4 py-5 text-xs text-muted">
                                    暂无已完成索引的文档
                                  </div>
                                ) : (
                                  <div className="overflow-hidden rounded-xl border border-line">
                                    <div className="hidden grid-cols-[minmax(0,1.6fr)_minmax(150px,1fr)_100px_130px] gap-3 bg-paper/70 px-4 py-2.5 text-[10px] font-bold uppercase tracking-[0.14em] text-muted sm:grid">
                                      <span>文档</span>
                                      <span>document_id</span>
                                      <span>chunk</span>
                                      <span className="text-right">创建时间</span>
                                    </div>
                                    <div className="divide-y divide-line">
                                      {knowledgeBase.documents.map((document) => (
                                        <div key={document.document_id} className="grid gap-2 px-4 py-3 sm:grid-cols-[minmax(0,1.6fr)_minmax(150px,1fr)_100px_130px] sm:items-center sm:gap-3">
                                          <div className="flex min-w-0 items-center gap-2">
                                            <FileText size={15} className="shrink-0 text-muted" />
                                            <span className="truncate text-sm font-semibold text-ink" title={document.title}>{document.title}</span>
                                          </div>
                                          <code className="break-all pl-6 font-mono text-[11px] text-muted sm:pl-0">{document.document_id}</code>
                                          <span className="pl-6 text-xs font-semibold text-ink/75 sm:pl-0">{document.chunk_count}</span>
                                          <span className="pl-6 text-xs text-muted sm:pl-0 sm:text-right">{formatDate(document.created_at)}</span>
                                        </div>
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
                </div>
              );
            })}
          </div>
        )}
      </section>

      {notice && (
        <div className="fixed bottom-5 right-5 z-50 rounded-xl border border-line bg-white px-4 py-3 text-sm font-semibold text-ink shadow-popover" role="status">
          {notice}
        </div>
      )}
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
  icon: React.ReactNode;
  title: string;
  description: string;
  action?: React.ReactNode;
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
